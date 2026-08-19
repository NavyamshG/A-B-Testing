"""
Shared design system for A/B Testing Studio.
Keeps a consistent look across all pages: colors, CSS, header component.
"""

import streamlit as st

PRIMARY = "#FF4B4B"
DARK = "#1E1E1E"
GREY = "#5E5E5E"
BG_CARD = "#f9f9f9"
GOOD = "#00CC96"
BAD = "#EF553B"
NEUTRAL = "#636EFA"
PURPLE = "#AB63FA"


def inject_base_css():
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] { font-size: 26px; }
        [data-testid="stMetricLabel"] { font-size: 13px; color: #777; }

        .sub-header {
            font-size: 1.25rem;
            color: #5E5E5E;
            margin-top: 2px;
            margin-bottom: 22px;
            font-weight: 400;
        }

        .description-box {
            background-color: #f9f9f9;
            padding: 18px 22px;
            border-radius: 10px;
            border-left: 5px solid #ff4b4b;
            margin-bottom: 22px;
        }

        .explanation-text {
            font-size: 0.97rem;
            line-height: 1.55;
            color: #333;
        }

        .callout {
            background-color: #e8f4f8;
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid #add8e6;
            margin-top: 16px;
        }

        .pitfall-card {
            background-color: #fff8f0;
            border: 1px solid #f0d9b5;
            border-left: 5px solid #f0a500;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }

        .nav-pill {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            background: #f0f0f0;
            font-size: 0.8rem;
            color: #555;
            margin-right: 6px;
        }

        section[data-testid="stSidebar"] { border-right: 1px solid #eee; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, emoji: str = ""):
    st.write(
        f"""
        <h1 style="font-size: 3rem; font-weight: 900; margin-bottom: 0px;">
            {emoji} <span style="
                background: linear-gradient(to right, #1E1E1E, #ff4b4b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">{title}</span>
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="sub-header">{subtitle}</p>', unsafe_allow_html=True)


def description_box(objective: str, description: str):
    st.markdown(
        f"""
        <div class="description-box">
            <strong>Objective:</strong> {objective}
            <hr style="margin: 15px 0; border: 0; border-top: 1px solid #ddd;">
            <div class="explanation-text">
                <strong>Description:</strong><br>{description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
