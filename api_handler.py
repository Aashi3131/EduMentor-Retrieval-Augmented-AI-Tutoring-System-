import json
import re
import html as html_lib
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from rag_retrieval import retrieve_rag_context
from web_evidence import fetch_web_evidence

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_VERIFIER_MODEL = "openai/gpt-5.4-mini"
MAX_VERIFY_MATERIAL_CHARS = 120_000

SYSTEM_INSTRUCTION = """You are EduMentor, a helpful course tutor.
You are given **retrieved excerpts** from the student's course PDFs (retrieval-augmented generation).
Base your answer primarily on those excerpts when they apply. If they do not cover the question, say so clearly.
When you use information from a source, name the PDF file."""

VERIFIER_PDF_SYSTEM = """You are an independent verifier (like a careful teaching assistant).
You receive: (1) extracted course materials, (2) the student's question, (3) a draft answer produced by another model.

Your job: judge how well the draft is supported by the materials only—not outside knowledge.

Reply with a single JSON object and nothing else. Keys:
- "verdict": exactly one of "verified", "mostly_correct", "uncertain", "unsupported"
- "reason": one short sentence for the user (plain text, no line breaks)

Definitions:
- "verified": The draft is accurate relative to the materials; no meaningful errors or hallucinations.
- "mostly_correct": Main ideas are right; only minor gaps, wording, or omissions.
- "uncertain": The materials are missing, ambiguous, or too thin to fully confirm important claims.
- "unsupported": The draft contradicts the materials, invents specifics, or claims things the text does not support.

If there are no materials, use "uncertain" unless the draft explicitly says the materials do not cover the topic."""

WEB_VERIFIER_SYSTEM = """You are an independent verifier checking a tutor's draft answer against **web evidence text** supplied below.

The web evidence is from automated search/snippets (may be incomplete or off-topic). Do not rely on facts that are not reflected in that evidence text.

You receive: (1) web evidence, (2) the student's question, (3) the draft answer.

Reply with a single JSON object and nothing else. Keys:
- "verdict": exactly one of "verified", "mostly_correct", "uncertain", "unsupported"
- "reason": one short sentence for the user (plain text, no line breaks)

Definitions:
- "verified": The draft's factual claims are well supported by or consistent with the web evidence; no major clashes.
- "mostly_correct": Broadly consistent with the evidence; only minor gaps or wording issues.
- "uncertain": The evidence is empty, too thin, or irrelevant to judge key claims—do not treat that as proof the draft is wrong.
- "unsupported": The draft makes factual claims that clearly contradict the web evidence or assert specifics the evidence does not support.

If the evidence explicitly says it could not be retrieved, prefer "uncertain" unless the draft is clearly non-factual or purely hedged."""

BADGE_META = {
    "verified": ("#198754", "Verified"),
    "mostly_correct": ("#0d6efd", "Mostly correct"),
    "uncertain": ("#fd7e14", "Uncertain"),
    "unsupported": ("#dc3545", "Not supported"),
    "skipped_error": ("#6c757d", "Verification skipped"),
    "verification_failed": ("#6c757d", "Verification unavailable"),
}


def _last_user_question(messages: list) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _materials_block(kb_docs: list) -> str:
    blocks = []
    for d in kb_docs:
        blocks.append(f"### {d['name']}\n{d['text']}")
    text = "\n\n".join(blocks) if blocks else "(No PDFs uploaded for this session.)"
    if len(text) > MAX_VERIFY_MATERIAL_CHARS:
        text = text[:MAX_VERIFY_MATERIAL_CHARS] + "\n\n[...truncated for verifier context length...]"
    return text


def _parse_verifier_json(raw: str) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    data = None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    if verdict not in ("verified", "mostly_correct", "uncertain", "unsupported"):
        return None
    reason = (data.get("reason") or "").strip()
    if not reason:
        reason = "No reason provided."
    return {"verdict": verdict, "reason": reason}


def _openrouter_verify_single(
    api_key: str,
    verifier_model: str,
    system_prompt: str,
    user_payload: str,
    x_title: str,
) -> dict:
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/EduMentor-local",
            "X-Title": x_title,
        },
    )
    try:
        resp = client.chat.completions.create(
            model=verifier_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.15,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _parse_verifier_json(raw)
        if parsed:
            return parsed
        return {
            "verdict": "verification_failed",
            "reason": "Could not parse verifier output.",
            "raw_excerpt": raw[:400],
        }
    except Exception as e:
        return {
            "verdict": "verification_failed",
            "reason": f"Verifier request failed: {e}",
        }


def verify_tutor_response(
    api_key: str,
    user_question: str,
    draft_answer: str,
    kb_docs: list,
    verifier_model: str = DEFAULT_VERIFIER_MODEL,
) -> dict:
    """Dual verification: PDF/course materials + web evidence, each via OpenRouter."""
    err_prefix = "**OpenRouter error:**"
    if draft_answer.strip().startswith(err_prefix):
        skipped = {"verdict": "skipped_error", "reason": "Primary model returned an error; nothing to verify."}
        return {"pdf": skipped, "web": skipped}

    materials = _materials_block(kb_docs)
    pdf_payload = (
        "--- Course materials (full PDF text for this session) ---\n\n"
        f"{materials}\n\n"
        "--- Student question ---\n\n"
        f"{user_question}\n\n"
        "--- Draft answer (from another model) ---\n\n"
        f"{draft_answer}"
    )

    web_evidence = fetch_web_evidence(user_question)
    if not (web_evidence or "").strip():
        web_evidence = (
            "(No web evidence could be retrieved for this query. "
            "Treat substantive factual checks as uncertain.)"
        )
    web_payload = (
        "--- Web evidence (snippets; may be incomplete) ---\n\n"
        f"{web_evidence}\n\n"
        "--- Student question ---\n\n"
        f"{user_question}\n\n"
        "--- Draft answer (from another model) ---\n\n"
        f"{draft_answer}"
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_pdf = pool.submit(
            _openrouter_verify_single,
            api_key,
            verifier_model,
            VERIFIER_PDF_SYSTEM,
            pdf_payload,
            "EduMentor-Verifier-PDF",
        )
        fut_web = pool.submit(
            _openrouter_verify_single,
            api_key,
            verifier_model,
            WEB_VERIFIER_SYSTEM,
            web_payload,
            "EduMentor-Verifier-Web",
        )
        pdf_v = fut_pdf.result()
        web_v = fut_web.result()

    return {"pdf": pdf_v, "web": web_v}


def _format_one_badge(verification: dict, channel_label: str) -> str:
    verdict = verification.get("verdict") or "verification_failed"
    reason = verification.get("reason") or ""
    color, label = BADGE_META.get(verdict, BADGE_META["verification_failed"])
    safe_reason = html_lib.escape(reason, quote=True)
    return (
        f'<p style="margin:0 0 0.45rem 0;">'
        f'<span style="color:#495057;font-size:0.72rem;font-weight:700;margin-right:0.35rem;">'
        f"{html_lib.escape(channel_label)}</span> "
        f'<span title="{safe_reason}" style="display:inline-block;padding:4px 12px;'
        f"border-radius:999px;background:{color};color:#fff;font-size:0.82rem;"
        f'font-weight:600;letter-spacing:0.02em;">{html_lib.escape(label)}</span> '
        f'<span style="color:#6c757d;font-size:0.8rem;margin-left:0.5rem;">{html_lib.escape(reason)}</span>'
        f"</p>"
    )


def format_verification_badge_markdown(verification: dict | None) -> str:
    """HTML badge rows for Streamlit markdown (unsafe_allow_html). Supports dual PDF + web."""
    if not verification:
        return ""
    if isinstance(verification.get("pdf"), dict) and isinstance(verification.get("web"), dict):
        return _format_one_badge(verification["pdf"], "PDF") + _format_one_badge(verification["web"], "Web")
    verdict = verification.get("verdict") or "verification_failed"
    reason = verification.get("reason") or ""
    color, label = BADGE_META.get(verdict, BADGE_META["verification_failed"])
    safe_reason = html_lib.escape(reason, quote=True)
    return (
        f'<p style="margin:0 0 0.6rem 0;">'
        f'<span title="{safe_reason}" style="display:inline-block;padding:4px 12px;'
        f"border-radius:999px;background:{color};color:#fff;font-size:0.82rem;"
        f'font-weight:600;letter-spacing:0.02em;">{html_lib.escape(label)}</span> '
        f'<span style="color:#6c757d;font-size:0.8rem;margin-left:0.5rem;">{html_lib.escape(reason)}</span>'
        f"</p>"
    )


def format_assistant_display_markdown(content: str, verification: dict | None) -> str:
    badge = format_verification_badge_markdown(verification)
    if badge:
        return f"{badge}\n{content}"
    return content


def send_query_get_response(api_key: str, messages: list, kb_docs: list, model: str) -> str:
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/EduMentor-local",
            "X-Title": "EduMentor",
        },
    )

    q = _last_user_question(messages)
    if kb_docs:
        rag = retrieve_rag_context(kb_docs, q, api_key)
        if rag.strip():
            materials = (
                "--- Retrieved excerpts (RAG from your PDFs; most relevant chunks first) ---\n\n" + rag
            )
        else:
            materials = (
                "(PDFs are attached but no excerpts could be retrieved—try rephrasing or check PDF text extraction.)"
            )
    else:
        materials = "(No PDFs uploaded for this session.)"

    openai_messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "system", "content": materials},
    ]
    for m in messages:
        role = m["role"]
        if role not in ("user", "assistant"):
            continue
        openai_messages.append({"role": role, "content": m["content"]})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=0.35,
            max_tokens=2048,
        )
        choice = resp.choices[0].message
        return (choice.content or "").strip() or "(Empty reply from model.)"
    except Exception as e:
        return (
            f"**OpenRouter error:** `{e}`\n\n"
            "Check your API key at https://openrouter.ai/keys and pick a **current** model id from "
            "https://openrouter.ai/models (404 “No endpoints found” usually means the slug was retired)."
        )
