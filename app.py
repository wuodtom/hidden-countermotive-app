from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Hidden Countermotive Interpreter v2",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    h1, h2, h3, h4 {
        color: #f8fafc;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="slider"] {
        padding-top: 0.35rem;
        padding-bottom: 0.65rem;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        padding: 14px 16px;
        border-radius: 14px;
        margin-bottom: 12px;
    }
    .soft-box {
        background: #111827;
        border: 1px solid #1f2937;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 12px;
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 6px 4px 0;
        border-radius: 999px;
        background: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


LATENT_LABELS = {
    "identity": "Identity construction",
    "status": "Status signaling",
    "belonging": "Belonging / tribe alignment",
    "aspiration": "Aspiration / future self",
    "morality": "Moral self-image",
    "control": "Control / certainty seeking",
    "narrative": "Transformation narrative",
    "functional": "Functional utility",
}

LATENT_ORDER = [
    "identity", "status", "belonging", "aspiration",
    "morality", "control", "narrative", "functional"
]

FEATURE_LABELS = {
    "price_premium": "Price premium",
    "social_visibility": "Social visibility",
    "identity_relevance": "Identity relevance",
    "exclusivity": "Exclusivity",
    "tribe_strength": "Tribe / subculture strength",
    "moral_framing": "Moral framing",
    "transformation_promise": "Transformation promise",
    "functional_necessity": "Functional necessity",
    "repetition": "Repetition / habit strength",
    "public_display": "Public display value",
}

SOCIAL_PLATFORM_PROFILES = {
    "General": {"social_visibility": 0, "public_display": 0, "status": 0.00, "belonging": 0.00, "narrative": 0.00},
    "WhatsApp Status": {"social_visibility": 1, "public_display": 1, "status": 0.03, "belonging": 0.05, "narrative": 0.03},
    "Instagram": {"social_visibility": 2, "public_display": 2, "status": 0.08, "belonging": 0.03, "narrative": 0.05},
    "Facebook": {"social_visibility": 1, "public_display": 1, "status": 0.03, "belonging": 0.04, "narrative": 0.03},
    "LinkedIn": {"social_visibility": 2, "public_display": 2, "status": 0.07, "belonging": 0.02, "narrative": 0.04},
}

DOMAIN_PRESETS = {
    "Balanced neutral": {
        "price_premium": 5, "social_visibility": 5, "identity_relevance": 5, "exclusivity": 5,
        "tribe_strength": 5, "moral_framing": 5, "transformation_promise": 5,
        "functional_necessity": 5, "repetition": 5, "public_display": 5,
    },
    "Luxury purchase": {
        "price_premium": 9, "social_visibility": 6, "identity_relevance": 8, "exclusivity": 8,
        "tribe_strength": 5, "moral_framing": 3, "transformation_promise": 7,
        "functional_necessity": 3, "repetition": 6, "public_display": 7,
    },
    "Fitness post": {
        "price_premium": 1, "social_visibility": 8, "identity_relevance": 9, "exclusivity": 3,
        "tribe_strength": 7, "moral_framing": 5, "transformation_promise": 8,
        "functional_necessity": 2, "repetition": 7, "public_display": 9,
    },
    "LinkedIn achievement post": {
        "price_premium": 1, "social_visibility": 8, "identity_relevance": 8, "exclusivity": 4,
        "tribe_strength": 6, "moral_framing": 4, "transformation_promise": 6,
        "functional_necessity": 2, "repetition": 6, "public_display": 9,
    },
    "Paid online course": {
        "price_premium": 7, "social_visibility": 4, "identity_relevance": 8, "exclusivity": 6,
        "tribe_strength": 7, "moral_framing": 2, "transformation_promise": 9,
        "functional_necessity": 3, "repetition": 5, "public_display": 4,
    },
    "Relationship signaling": {
        "price_premium": 4, "social_visibility": 8, "identity_relevance": 8, "exclusivity": 5,
        "tribe_strength": 6, "moral_framing": 4, "transformation_promise": 5,
        "functional_necessity": 2, "repetition": 5, "public_display": 9,
    },
}

WORKED_EXAMPLES = {
    "Premium skincare purchase": {
        "domain": "Consumption / product purchase",
        "platform": "General",
        "observed": "A person buys a $200 skincare routine.",
        "surface": "They want healthier, clearer skin.",
        "context": "Premium product, repeat use, moderate visibility, identity-linked self-care behavior.",
        "features": {"price_premium": 9, "social_visibility": 6, "identity_relevance": 8, "exclusivity": 7,
                     "tribe_strength": 5, "moral_framing": 4, "transformation_promise": 8,
                     "functional_necessity": 3, "repetition": 8, "public_display": 6},
    },
    "Instagram gym transformation post": {
        "domain": "Social media behavior",
        "platform": "Instagram",
        "observed": "A person posts a gym mirror transformation photo on Instagram.",
        "surface": "They want to share progress and motivate others.",
        "context": "Highly visible body display, identity performance, admiration seeking, narrative arc.",
        "features": {"price_premium": 1, "social_visibility": 9, "identity_relevance": 9, "exclusivity": 3,
                     "tribe_strength": 7, "moral_framing": 5, "transformation_promise": 8,
                     "functional_necessity": 1, "repetition": 7, "public_display": 10},
    },
    "LinkedIn certificate post": {
        "domain": "Social media behavior",
        "platform": "LinkedIn",
        "observed": "A person posts a new certificate on LinkedIn with a reflection caption.",
        "surface": "They want to share learning progress and professional growth.",
        "context": "Professional signaling, competence display, career identity building, visible achievement framing.",
        "features": {"price_premium": 2, "social_visibility": 8, "identity_relevance": 8, "exclusivity": 4,
                     "tribe_strength": 6, "moral_framing": 4, "transformation_promise": 6,
                     "functional_necessity": 2, "repetition": 6, "public_display": 9},
    },
    "WhatsApp status motivational quote": {
        "domain": "Social media behavior",
        "platform": "WhatsApp Status",
        "observed": "A person posts a motivational quote on WhatsApp Status after a setback.",
        "surface": "They want to encourage themselves and others.",
        "context": "Semi-private social signaling, emotional repositioning, identity repair, resilience narrative.",
        "features": {"price_premium": 0, "social_visibility": 6, "identity_relevance": 7, "exclusivity": 1,
                     "tribe_strength": 5, "moral_framing": 5, "transformation_promise": 6,
                     "functional_necessity": 1, "repetition": 6, "public_display": 7},
    },
}

TEXT_CUES = {
    "price_premium": {
        2: ["premium", "luxury", "expensive", "$", "elite", "high-priced", "exclusive", "designer"],
        1: ["paid", "costly", "subscription", "membership"]
    },
    "social_visibility": {
        3: ["post", "posted", "public", "viral", "followers", "audience", "people can see", "announcement"],
        2: ["instagram", "linkedin", "facebook", "status", "share", "shared", "story", "caption"]
    },
    "identity_relevance": {
        3: ["identity", "who i am", "who they are", "becoming", "discipline", "disciplined", "professional", "serious"],
        2: ["self-image", "brand", "image", "character", "type of person", "version of myself"]
    },
    "exclusivity": {
        2: ["exclusive", "members-only", "elite", "invite-only", "rare", "premium access", "limited"],
        1: ["select", "top-tier", "high-end"]
    },
    "tribe_strength": {
        2: ["community", "tribe", "group", "belonging", "like-minded", "network", "club", "fitfam"],
        1: ["circle", "people like me", "scene", "subculture"]
    },
    "moral_framing": {
        2: ["clean", "pure", "healthy", "good", "better person", "authentic", "ethical", "organic"],
        1: ["right thing", "worthy", "responsible", "values"]
    },
    "transformation_promise": {
        3: ["transform", "transformation", "glow up", "level up", "upgrade", "new me", "future self"],
        2: ["progress", "journey", "growth", "evolve", "become", "change my life"]
    },
    "functional_necessity": {
        3: ["need", "necessary", "required", "must", "practical", "solve", "fix"],
        2: ["useful", "function", "works", "effective", "efficient"]
    },
    "repetition": {
        2: ["routine", "every day", "daily", "weekly", "consistent", "habit", "ongoing", "streak"],
        1: ["regularly", "again", "still going"]
    },
    "public_display": {
        3: ["show", "display", "certificate", "badge", "photo", "mirror selfie", "announcement", "achievement"],
        2: ["proof", "evidence", "visible", "recognition", "signal"]
    }
}


@dataclass
class InputFeatures:
    price_premium: float
    social_visibility: float
    identity_relevance: float
    exclusivity: float
    tribe_strength: float
    moral_framing: float
    transformation_promise: float
    functional_necessity: float
    repetition: float
    public_display: float


def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def initialize_state() -> None:
    defaults = {
        "domain": "Consumption / product purchase",
        "platform": "General",
        "observed": "",
        "surface": "",
        "context": "",

        "domain_widget": "Consumption / product purchase",
        "platform_widget": "General",
        "observed_widget": "",
        "surface_widget": "",
        "context_widget": "",

        "price_premium": 5.0,
        "social_visibility": 5.0,
        "identity_relevance": 5.0,
        "exclusivity": 5.0,
        "tribe_strength": 5.0,
        "moral_framing": 5.0,
        "transformation_promise": 5.0,
        "functional_necessity": 5.0,
        "repetition": 5.0,
        "public_display": 5.0,
        "last_analysis": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sync_widget_state_to_model_state() -> None:
    st.session_state.domain = st.session_state.domain_widget
    st.session_state.platform = st.session_state.platform_widget
    st.session_state.observed = st.session_state.observed_widget
    st.session_state.surface = st.session_state.surface_widget
    st.session_state.context = st.session_state.context_widget


def features_from_state() -> InputFeatures:
    return InputFeatures(
        price_premium=float(st.session_state.price_premium),
        social_visibility=float(st.session_state.social_visibility),
        identity_relevance=float(st.session_state.identity_relevance),
        exclusivity=float(st.session_state.exclusivity),
        tribe_strength=float(st.session_state.tribe_strength),
        moral_framing=float(st.session_state.moral_framing),
        transformation_promise=float(st.session_state.transformation_promise),
        functional_necessity=float(st.session_state.functional_necessity),
        repetition=float(st.session_state.repetition),
        public_display=float(st.session_state.public_display),
    )


def apply_preset(name: str) -> None:
    preset = DOMAIN_PRESETS[name]
    for k, v in preset.items():
        st.session_state[k] = float(v)
    st.session_state.last_analysis = None


def load_example(name: str) -> None:
    ex = WORKED_EXAMPLES[name]

    st.session_state.domain = ex["domain"]
    st.session_state.platform = ex["platform"]
    st.session_state.observed = ex["observed"]
    st.session_state.surface = ex["surface"]
    st.session_state.context = ex["context"]

    st.session_state.domain_widget = ex["domain"]
    st.session_state.platform_widget = ex["platform"]
    st.session_state.observed_widget = ex["observed"]
    st.session_state.surface_widget = ex["surface"]
    st.session_state.context_widget = ex["context"]

    for k, v in ex["features"].items():
        st.session_state[k] = float(v)

    st.session_state.last_analysis = None


def normalize_features(features: InputFeatures) -> np.ndarray:
    return np.array(list(asdict(features).values()), dtype=float) / 10.0


def auto_score_from_text(observed: str, surface: str, context: str, platform: str) -> Dict[str, float]:
    text = f"{observed} {surface} {context} {platform}".lower()
    scores = {k: 0.0 for k in FEATURE_LABELS}

    for feature, bundles in TEXT_CUES.items():
        for weight, phrases in bundles.items():
            for phrase in phrases:
                if phrase in text:
                    scores[feature] += weight

    if platform in SOCIAL_PLATFORM_PROFILES:
        scores["social_visibility"] += SOCIAL_PLATFORM_PROFILES[platform]["social_visibility"] * 1.8
        scores["public_display"] += SOCIAL_PLATFORM_PROFILES[platform]["public_display"] * 1.8

    for key in scores:
        scores[key] = clamp(2 + scores[key], 0, 10)

    return scores


def blend_features(manual: InputFeatures, auto_scores: Dict[str, float], auto_weight: float) -> InputFeatures:
    manual_dict = asdict(manual)
    blended = {}
    for key, manual_value in manual_dict.items():
        blended[key] = (1 - auto_weight) * manual_value + auto_weight * auto_scores[key]
    return InputFeatures(**blended)


def compute_hidden_drivers(features: InputFeatures, platform: str = "General") -> Dict[str, float]:
    x = normalize_features(features)

    weights = {
        "identity":   np.array([0.08, 0.07, 0.30, 0.07, 0.06, 0.05, 0.22, -0.10, 0.05, 0.10]),
        "status":     np.array([0.14, 0.20, 0.11, 0.15, 0.05, 0.00, 0.06, -0.05, 0.01, 0.23]),
        "belonging":  np.array([0.03, 0.10, 0.10, 0.06, 0.35, 0.02, 0.09, -0.04, 0.07, 0.12]),
        "aspiration": np.array([0.06, 0.05, 0.18, 0.06, 0.03, 0.02, 0.38, -0.08, 0.05, 0.05]),
        "morality":   np.array([0.01, 0.02, 0.07, 0.02, 0.04, 0.52, 0.08, -0.05, 0.07, 0.02]),
        "control":    np.array([0.02, 0.02, 0.09, 0.02, 0.03, 0.11, 0.14, 0.24, 0.24, 0.02]),
        "narrative":  np.array([0.03, 0.06, 0.15, 0.03, 0.05, 0.03, 0.40, -0.05, 0.10, 0.07]),
        "functional": np.array([0.03, 0.00, 0.03, 0.01, 0.00, 0.00, 0.03, 0.78, 0.10, 0.02]),
    }
    base_bias = {
        "identity": 0.06, "status": 0.05, "belonging": 0.05, "aspiration": 0.06,
        "morality": 0.04, "control": 0.05, "narrative": 0.05, "functional": 0.03,
    }

    scores = {}
    for key, w in weights.items():
        raw = float(np.dot(w, x) + base_bias[key])
        scores[key] = clamp01(raw)

    profile = SOCIAL_PLATFORM_PROFILES.get(platform, SOCIAL_PLATFORM_PROFILES["General"])
    scores["status"] = clamp01(scores["status"] + profile["status"])
    scores["belonging"] = clamp01(scores["belonging"] + profile["belonging"])
    scores["narrative"] = clamp01(scores["narrative"] + profile["narrative"])

    return scores


def compute_hcr(scores: Dict[str, float]) -> float:
    latent_keys = ["identity", "status", "belonging", "aspiration", "morality", "control", "narrative"]
    latent_mean = float(np.mean([scores[k] for k in latent_keys]))
    functional = max(scores["functional"], 0.05)
    return latent_mean / functional


def classify_hcr(hcr: float) -> str:
    if hcr < 1.0:
        return "Mostly functional"
    if hcr < 1.5:
        return "Function and hidden drivers are mixed"
    if hcr < 2.2:
        return "Hidden countermotives are strong"
    return "Hidden countermotives dominate"


def top_latent_drivers(scores: Dict[str, float], n: int = 3) -> List[Tuple[str, float]]:
    latent_keys = ["identity", "status", "belonging", "aspiration", "morality", "control", "narrative"]
    return sorted(((k, scores[k]) for k in latent_keys), key=lambda kv: kv[1], reverse=True)[:n]


def calculate_confidence(manual: InputFeatures, auto_scores: Dict[str, float], observed: str, surface: str, context: str) -> Tuple[float, str]:
    manual_dict = asdict(manual)
    diffs = [abs(manual_dict[k] - auto_scores[k]) / 10 for k in manual_dict]
    agreement = 1 - float(np.mean(diffs))

    text_len = len((observed + " " + surface + " " + context).split())
    richness = min(text_len / 40, 1.0)

    spread = np.std(list(manual_dict.values())) / 3
    specificity = min(spread, 1.0)

    confidence = 0.50 * agreement + 0.25 * richness + 0.25 * specificity
    confidence = clamp01(confidence)

    if confidence >= 0.78:
        label = "High"
    elif confidence >= 0.58:
        label = "Moderate"
    else:
        label = "Low"
    return confidence, label


def generate_best_structure(scores: Dict[str, float], surface: str, domain: str, platform: str) -> str:
    top = top_latent_drivers(scores, 3)
    labels = [LATENT_LABELS[k] for k, _ in top]
    platform_phrase = ""
    if domain.lower().startswith("social media"):
        platform_phrase = f" On {platform}, audience visibility makes signaling and identity management more active."
    return (
        f'This phenomenon is framed on the surface as: "{surface}". '
        f"But the stronger motivational structure underneath is {labels[0]}, reinforced by {labels[1]} and {labels[2]}.{platform_phrase}"
    )


def generate_plain_translation(scores: Dict[str, float], surface: str, platform: str) -> str:
    top = top_latent_drivers(scores, 2)
    first, second = [LATENT_LABELS[k].lower() for k, _ in top]
    social_hint = ""
    if platform in ("Instagram", "LinkedIn", "Facebook", "WhatsApp Status"):
        social_hint = " The behavior is also doing social work by shaping how other people read the person."
    return (
        f"On the surface, this looks like: {surface.lower()}. "
        f"But a cleaner interpretation is that the behavior is also serving {first} and {second}.{social_hint}"
    )


def generate_driver_notes(scores: Dict[str, float]) -> Dict[str, str]:
    notes = {}
    for key in LATENT_ORDER:
        val = scores[key]
        label = LATENT_LABELS[key]
        if val >= 0.75:
            notes[key] = f"{label} is a dominant force here."
        elif val >= 0.55:
            notes[key] = f"{label} is clearly active."
        elif val >= 0.35:
            notes[key] = f"{label} is present but not dominant."
        else:
            notes[key] = f"{label} is relatively weak in this case."
    return notes


def result_dataframe(scores: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "Driver": [LATENT_LABELS[k] for k in LATENT_ORDER],
        "Score": [round(scores[k], 3) for k in LATENT_ORDER],
    }).sort_values("Score", ascending=False)


def feature_dataframe(manual: InputFeatures, auto_scores: Dict[str, float], blended: InputFeatures) -> pd.DataFrame:
    manual_dict = asdict(manual)
    blended_dict = asdict(blended)
    return pd.DataFrame({
        "Feature": [FEATURE_LABELS[k] for k in manual_dict],
        "Manual score": [round(manual_dict[k], 2) for k in manual_dict],
        "Auto text score": [round(auto_scores[k], 2) for k in manual_dict],
        "Blended score": [round(blended_dict[k], 2) for k in manual_dict],
    })


def make_bar_chart(scores: Dict[str, float]):
    labels = [LATENT_LABELS[k] for k in LATENT_ORDER]
    values = [scores[k] for k in LATENT_ORDER]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Hidden Countermotive Profile")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def make_radar_chart(scores: Dict[str, float]):
    keys = LATENT_ORDER
    labels = [LATENT_LABELS[k] for k in keys]
    values = [scores[k] for k in keys]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Motive Balance Map", pad=20)
    fig.tight_layout()
    return fig


def make_feature_chart(manual: InputFeatures, auto_scores: Dict[str, float], blended: InputFeatures):
    manual_dict = asdict(manual)
    blended_dict = asdict(blended)
    labels = [FEATURE_LABELS[k] for k in manual_dict]
    manual_vals = [manual_dict[k] for k in manual_dict]
    auto_vals = [auto_scores[k] for k in manual_dict]
    blended_vals = [blended_dict[k] for k in manual_dict]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width, manual_vals, width, label="Manual")
    ax.bar(x, auto_vals, width, label="Auto text")
    ax.bar(x + width, blended_vals, width, label="Blended")
    ax.set_ylim(0, 10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Feature score")
    ax.set_title("Manual vs auto vs blended feature scoring")
    ax.legend()
    fig.tight_layout()
    return fig


def create_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def create_pdf_report(summary: Dict, scores_df: pd.DataFrame, features_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with PdfPages(output) as pdf:
        fig1 = plt.figure(figsize=(8.27, 11.69))
        fig1.text(0.06, 0.95, "Hidden Countermotive Interpreter Report", fontsize=16, weight="bold")
        y = 0.90
        lines = [
            f"Domain: {summary['domain']}",
            f"Platform: {summary['platform']}",
            f"Observed phenomenon: {summary['observed']}",
            f"Surface function: {summary['surface']}",
            f"HCR: {summary['hcr']:.2f} ({summary['hcr_label']})",
            f"Confidence: {summary['confidence']:.2f} ({summary['confidence_label']})",
            "",
            "Best-structure explanation:",
            summary['best_structure'],
            "",
            "Plain-language translation:",
            summary['plain_translation'],
            "",
            "Context:",
            summary['context'] or "No extra context provided.",
        ]
        for line in lines:
            fig1.text(0.06, y, line, fontsize=10, wrap=True)
            y -= 0.035 if len(line) < 110 else 0.055
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = make_bar_chart(summary["scores"])
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        fig3 = make_feature_chart(summary["manual_features"], summary["auto_scores"], summary["blended_features"])
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        fig4 = plt.figure(figsize=(8.27, 11.69))
        fig4.text(0.06, 0.96, "Driver Scores", fontsize=13, weight="bold")
        ax1 = fig4.add_axes([0.05, 0.58, 0.90, 0.30])
        ax1.axis("off")
        table1 = ax1.table(cellText=scores_df.values, colLabels=scores_df.columns, loc="center")
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1, 1.5)

        fig4.text(0.06, 0.50, "Feature Scores", fontsize=13, weight="bold")
        ax2 = fig4.add_axes([0.05, 0.05, 0.90, 0.40])
        ax2.axis("off")
        table2 = ax2.table(cellText=features_df.values, colLabels=features_df.columns, loc="center")
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(1, 1.3)
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    output.seek(0)
    return output.read()


initialize_state()

st.title("🧠 Hidden Countermotive Interpreter v2")
st.caption("Observed phenomenon → stated surface function → inferred hidden driver")

with st.expander("What changed in v2", expanded=True):
    st.markdown("""
- text-based auto-scoring from the user’s written description  
- CSV and PDF export  
- domain presets  
- confidence score  
- manual vs auto vs blended feature comparison
""")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("1) Describe the phenomenon")
    st.selectbox(
        "Domain",
        ["Consumption / product purchase", "Education / information product", "Fitness / self-improvement",
         "Professional signaling", "Social media behavior", "Community / belonging", "Relationship signaling", "Custom"],
        key="domain_widget",
    )
    st.selectbox(
        "Platform / environment",
        ["General", "WhatsApp Status", "Instagram", "Facebook", "LinkedIn"],
        key="platform_widget"
    )

    st.text_area(
        "Observed phenomenon",
        key="observed_widget",
        height=120,
        placeholder="Example: A person posts a certificate on LinkedIn after completing a course.",
    )
    st.text_area(
        "Stated surface function",
        key="surface_widget",
        height=100,
        placeholder="Example: They want to share learning progress and professional growth.",
    )
    st.text_area(
        "Optional context notes",
        key="context_widget",
        height=100,
        placeholder="Add any extra cues: premium pricing, audience, emotional tone, repetition, identity language, etc.",
    )

    st.subheader("2) Use a preset or score manually")
    preset_name = st.selectbox("Domain preset", list(DOMAIN_PRESETS.keys()))
    pc1, pc2 = st.columns(2)
    with pc1:
        if st.button("Apply preset", use_container_width=True):
            apply_preset(preset_name)
            st.rerun()
    with pc2:
        if st.button("Reset sliders", use_container_width=True):
            apply_preset("Balanced neutral")
            st.rerun()

    st.caption("Manual sliders")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Price premium", 0.0, 10.0, key="price_premium")
        st.slider("Social visibility", 0.0, 10.0, key="social_visibility")
        st.slider("Identity relevance", 0.0, 10.0, key="identity_relevance")
        st.slider("Exclusivity", 0.0, 10.0, key="exclusivity")
        st.slider("Tribe / subculture strength", 0.0, 10.0, key="tribe_strength")
    with c2:
        st.slider("Moral framing", 0.0, 10.0, key="moral_framing")
        st.slider("Transformation promise", 0.0, 10.0, key="transformation_promise")
        st.slider("Functional necessity", 0.0, 10.0, key="functional_necessity")
        st.slider("Repetition / habit strength", 0.0, 10.0, key="repetition")
        st.slider("Public display value", 0.0, 10.0, key="public_display")

    st.subheader("3) Blending")
    auto_weight = st.slider(
        "How much should text auto-scoring influence the final model?",
        min_value=0.0, max_value=1.0, value=0.35, step=0.05,
        help="0 = manual only. 1 = text auto-scoring only."
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        analyze = st.button("Analyze phenomenon", use_container_width=True)
    with bc2:
        clear = st.button("Clear text fields", use_container_width=True)

    if clear:
        st.session_state.observed = ""
        st.session_state.surface = ""
        st.session_state.context = ""

        st.session_state.observed_widget = ""
        st.session_state.surface_widget = ""
        st.session_state.context_widget = ""

        st.session_state.last_analysis = None
        st.rerun()

with right:
    st.subheader("Worked examples")
    st.markdown('<div class="small-note">Load any example directly into the model.</div>', unsafe_allow_html=True)
    for name in WORKED_EXAMPLES:
        if st.button(name, key=f"ex_{name}", use_container_width=True):
            load_example(name)
            st.rerun()

    st.markdown("---")
    st.subheader("Model flow")
    st.markdown("""
<div class="soft-box">
Observed phenomenon<br>
↓<br>
Stated surface function<br>
↓<br>
Text auto-scoring + manual scoring<br>
↓<br>
Blended feature profile<br>
↓<br>
Latent driver engine<br>
↓<br>
Best-structure explanation + confidence
</div>
""", unsafe_allow_html=True)

    st.subheader("Hidden drivers")
    for item in LATENT_ORDER:
        st.markdown(f'<span class="pill">{LATENT_LABELS[item]}</span>', unsafe_allow_html=True)

sync_widget_state_to_model_state()

should_run = analyze or bool(
    st.session_state.observed.strip() and
    st.session_state.surface.strip() and
    st.session_state.get("last_analysis") is None
)

if should_run:
    if not st.session_state.observed.strip() or not st.session_state.surface.strip():
        st.warning("Add both the observed phenomenon and stated surface function.")
    else:
        manual_features = features_from_state()
        auto_scores = auto_score_from_text(
            st.session_state.observed,
            st.session_state.surface,
            st.session_state.context,
            st.session_state.platform
        )
        blended_features = blend_features(manual_features, auto_scores, auto_weight)
        scores = compute_hidden_drivers(blended_features, platform=st.session_state.platform)
        hcr = compute_hcr(scores)
        hcr_label = classify_hcr(hcr)
        confidence, confidence_label = calculate_confidence(
            manual_features, auto_scores, st.session_state.observed, st.session_state.surface, st.session_state.context
        )
        best_structure = generate_best_structure(scores, st.session_state.surface, st.session_state.domain, st.session_state.platform)
        plain_translation = generate_plain_translation(scores, st.session_state.surface, st.session_state.platform)
        scores_df = result_dataframe(scores)
        features_df = feature_dataframe(manual_features, auto_scores, blended_features)

        summary = {
            "domain": st.session_state.domain,
            "platform": st.session_state.platform,
            "observed": st.session_state.observed,
            "surface": st.session_state.surface,
            "context": st.session_state.context,
            "hcr": hcr,
            "hcr_label": hcr_label,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "best_structure": best_structure,
            "plain_translation": plain_translation,
            "scores": scores,
            "manual_features": manual_features,
            "auto_scores": auto_scores,
            "blended_features": blended_features,
            "scores_df": scores_df,
            "features_df": features_df,
        }
        st.session_state.last_analysis = summary

if st.session_state.last_analysis:
    summary = st.session_state.last_analysis
    scores = summary["scores"]
    scores_df = summary["scores_df"]
    features_df = summary["features_df"]
    top3 = top_latent_drivers(scores)
    notes = generate_driver_notes(scores)

    st.markdown("---")
    st.subheader("4) Results")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Hidden Countermotive Ratio", f"{summary['hcr']:.2f}")
    with m2:
        st.metric("Model read", summary["hcr_label"])
    with m3:
        st.metric("Confidence", f"{summary['confidence']:.2f} · {summary['confidence_label']}")
    with m4:
        st.metric("Top hidden driver", LATENT_LABELS[top3[0][0]])

    a, b = st.columns([1.15, 0.85], gap="large")
    with a:
        st.markdown("### Best-structure explanation")
        st.markdown(f'<div class="soft-box">{summary["best_structure"]}</div>', unsafe_allow_html=True)
        st.markdown("### Plain-language translation")
        st.markdown(f'<div class="soft-box">{summary["plain_translation"]}</div>', unsafe_allow_html=True)
        st.markdown("### Driver table")
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

    with b:
        st.markdown("### Top 3 drivers")
        for k, val in top3:
            st.markdown(
                f'<div class="metric-card"><strong>{LATENT_LABELS[k]}</strong><br>Score: {val:.2f}<br>'
                f'<span class="small-note">{notes[k]}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("### Interpretation note")
        st.markdown(
            f"""
<div class="soft-box">
Surface claim: <strong>{summary["surface"]}</strong><br><br>
Observed phenomenon: <strong>{summary["observed"]}</strong><br><br>
Context: <span class="small-note">{summary["context"] or "No extra context provided."}</span><br><br>
Confidence explanation: <span class="small-note">Confidence rises when manual and auto text scores agree, the text is specific, and the feature profile is not flat.</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Hidden driver bar chart")
        st.pyplot(make_bar_chart(scores), clear_figure=True)
    with c2:
        st.markdown("### Motive balance radar chart")
        st.pyplot(make_radar_chart(scores), clear_figure=True)

    st.markdown("### Feature scoring comparison")
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    st.pyplot(
        make_feature_chart(summary["manual_features"], summary["auto_scores"], summary["blended_features"]),
        clear_figure=True
    )

    st.markdown("### Export")
    csv_bytes = create_csv_bytes(scores_df)
    pdf_bytes = create_pdf_report(summary, scores_df, features_df)

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download driver scores CSV",
            data=csv_bytes,
            file_name="hidden_countermotive_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="hidden_countermotive_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("### Model caveat")
    st.info(
        "This is an interpretive model. It does not prove intent. "
        "It estimates likely hidden drivers from visible patterns, text cues, context, and platform effects."
    )

with st.sidebar:
    st.header("Run locally")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

    st.header("Scoring guide")
    st.markdown("""
- **Price premium**: costly relative to alternatives  
- **Social visibility**: how visible it is to others  
- **Identity relevance**: how much it links to self-image  
- **Exclusivity**: prestige, rarity, access  
- **Tribe strength**: community or subculture pull  
- **Moral framing**: purity, virtue, disciplined-self language  
- **Transformation promise**: upgrade, glow-up, future self  
- **Functional necessity**: practical need  
- **Repetition**: recurring habit or routine  
- **Public display value**: performs well in front of others
""")

    st.header("Text auto-score hint")
    st.markdown("""
Text auto-scoring looks for cues like:
- **identity**: becoming, disciplined, professional
- **status**: premium, elite, public achievement
- **narrative**: journey, transformation, progress
- **morality**: clean, pure, ethical
- **function**: need, practical, useful
""")
