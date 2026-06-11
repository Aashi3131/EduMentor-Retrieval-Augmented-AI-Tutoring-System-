import html as html_lib
from datetime import datetime

import markdown2  # assuming markdown2 is installed

from api_handler import BADGE_META

def generate_html(messages):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_content = ""
    interaction_count = 0
    user_emoji = "User"  # Emoji for user
    assistant_emoji = "AI-Tutor"  # Emoji for assistant

    for message in messages:
        role_class = 'user' if message['role'] == 'user' else 'assistant'
        emoji = user_emoji if role_class == 'user' else assistant_emoji
        if role_class == 'user':
            interaction_count += 1

        badge_html = ""
        if role_class == 'assistant' and message.get('verification'):
            v = message['verification']

            def _one_row(verdict: str, reason: str, prefix: str) -> str:
                color, label = BADGE_META.get(verdict, BADGE_META['verification_failed'])
                return (
                    f"<p class='verify-row'><span class='verify-prefix'>{html_lib.escape(prefix)}</span> "
                    f"<span class='verify-badge' style='background:{color};'>"
                    f"{html_lib.escape(label)}</span> "
                    f"<span class='verify-reason'>{html_lib.escape(reason)}</span></p>"
                )

            if isinstance(v.get('pdf'), dict) and isinstance(v.get('web'), dict):
                pv, wv = v['pdf'], v['web']
                badge_html = (
                    _one_row(
                        pv.get('verdict') or 'verification_failed',
                        pv.get('reason') or '',
                        'PDF',
                    )
                    + _one_row(
                        wv.get('verdict') or 'verification_failed',
                        wv.get('reason') or '',
                        'Web',
                    )
                )
            else:
                verdict = v.get('verdict') or 'verification_failed'
                reason = v.get('reason') or ''
                color, label = BADGE_META.get(verdict, BADGE_META['verification_failed'])
                badge_html = (
                    f"<p class='verify-row'><span class='verify-badge' style='background:{color};'>"
                    f"{html_lib.escape(label)}</span> "
                    f"<span class='verify-reason'>{html_lib.escape(reason)}</span></p>"
                )

        formatted_content = markdown2.markdown(message['content']).replace('\\n', '<br>')
        chat_content += (
            f"<div class='message {role_class}'><span class='emoji'>{emoji}</span>"
            f"{badge_html}{formatted_content}</div>"
        )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EduMentor Chat History</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
        .container {{
            margin-top: 20px;
            background: white;
            border-radius: 0.25rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,.075);
        }}
        .header {{
            background-color: #30694b; /* A deep, elegant shade of green */
            color: white;
            padding: 1rem;
            text-align: center;
        }}
        .message {{
            padding: 0.5rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 10px;
            color: #000080; /* Navy */
        }}
        .user {{
            background-color: #e8f5e9; /* Light Emerald Green */
            align-self: flex-start;
            text-align: left;
        }}
        .assistant {{
            background-color: #d4edda;
        }}
        .footer {{
            background-color: #30694b; /* A deep, elegant shade of green */
            color: white;
            padding: 1rem;
            text-align: center;
        }}
        .timestamp, .interaction-count {{
            font-size: 0.875rem;
            color: #6c757d;
            text-align: center;
            padding: 0.5rem 0;
        }}
        .emoji {{
            font-size: 1.2em;
            margin-right: 0.5rem;
        }}
        .verify-row {{
            margin: 0 0 0.5rem 0;
        }}
        .verify-prefix {{
            color: #495057;
            font-size: 0.72rem;
            font-weight: 700;
            margin-right: 0.35rem;
        }}
        .verify-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            color: #fff;
            font-size: 0.82rem;
            font-weight: 600;
        }}
        .verify-reason {{
            color: #6c757d;
            font-size: 0.85rem;
            margin-left: 0.35rem;
        }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>EduMentor Chat History</h1>
                <div>Generated on: {timestamp}</div>
                <div>Total Q&A Interactions: {interaction_count}</div>
            </div>
            <div>
                {chat_content}
            </div>
            <div class="footer">
                This conversation is generated by EduMentor AI Tutor.
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
