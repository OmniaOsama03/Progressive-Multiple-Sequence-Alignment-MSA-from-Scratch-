import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from dpp import distance_matrix, progressive_align


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Progressive MSA Visualizer",
    page_icon="🧬",
    layout="wide"
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
    .hero-card {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #111827 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.28);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.3rem;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    .soft-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
        margin-bottom: 0.9rem;
    }

    .soft-card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.35rem;
    }

    .soft-card-text {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.55;
    }

    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.14);
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.2rem;
        margin-bottom: 0.65rem;
    }

    .seq-card {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: #020617;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.6rem;
        font-family: Consolas, Monaco, monospace;
        overflow-x: auto;
    }

    .seq-label {
        color: #93c5fd;
        font-weight: 700;
        margin-right: 0.6rem;
    }

    .seq-text {
        color: #e2e8f0;
        letter-spacing: 0.03em;
    }

    .gap {
        color: #fda4af;
        font-weight: 800;
    }

    .aa {
        color: #e2e8f0;
    }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.14);
        border: 1px solid rgba(59,130,246,0.22);
        color: #bfdbfe;
        font-size: 0.84rem;
        margin-right: 0.4rem;
        margin-bottom: 0.35rem;
    }

    .tiny-note {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .result-box {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def parse_sequences(text: str):
    return [line.strip().upper() for line in text.splitlines() if line.strip()]

def validate_sequences(sequences):
    allowed = set("ACDEFGHIKLMNPQRSTVWY-")
    if len(sequences) == 0:
        return False, "Please enter at least one sequence."

    for seq in sequences:
        for ch in seq:
            if ch not in allowed:
                return False, (
                    f"Invalid character '{ch}' found. "
                    "Use only the 20 standard amino acids and optional '-' gaps."
                )
    return True, ""

def plot_distance_heatmap(D: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(D.values, aspect="auto")

    ax.set_xticks(range(len(D.columns)))
    ax.set_xticklabels([f"S{i+1}" for i in range(len(D.columns))], rotation=45, ha="right")
    ax.set_yticks(range(len(D.index)))
    ax.set_yticklabels([f"S{i+1}" for i in range(len(D.index))])

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("Pairwise Distance Matrix", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def compute_linkage_matrix(D: pd.DataFrame):
    condensed = squareform(D.values)
    Z = linkage(condensed, method="average")
    return Z

def plot_dendrogram_from_Z(Z, n_labels):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    dendrogram(Z, labels=[f"S{i+1}" for i in range(n_labels)], ax=ax)
    ax.set_title("Guide Tree (UPGMA / Average Linkage)", pad=12)
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig

def build_alignment_table(aligned_sequences):
    return pd.DataFrame({
        "Sequence #": [f"S{i+1}" for i in range(len(aligned_sequences))],
        "Aligned Sequence": aligned_sequences,
        "Aligned Length": [len(seq) for seq in aligned_sequences],
        "Recovered Original": [seq.replace("-", "") for seq in aligned_sequences]
    })

def style_alignment(seq: str):
    pieces = []
    for ch in seq:
        if ch == "-":
            pieces.append(f"<span class='gap'>{ch}</span>")
        else:
            pieces.append(f"<span class='aa'>{ch}</span>")
    return "".join(pieces)

def build_fasta_output(aligned_sequences):
    lines = []
    for i, seq in enumerate(aligned_sequences, start=1):
        lines.append(f">Sequence_{i}")
        lines.append(seq)
    return "\n".join(lines)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
sample_sequences = [
    "MKTAYIAKQRQISFVKSHFSRQ",
    "MKTAYIAKQRQISFVKSHFSRL",
    "MKTAYIAQRQISFVKSHFSRQ",
    "MKTAIAKQRQISFVKSHFSR"
]

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    gap_penalty = st.number_input("Gap penalty", value=-1, step=1)

    if "sequence_input" not in st.session_state:
        st.session_state.sequence_input = ""

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Load Sample", use_container_width=True):
            st.session_state.sequence_input = "\n".join(sample_sequences)
    with col_b:
        if st.button("Clear", use_container_width=True):
            st.session_state.sequence_input = ""

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "<div class='tiny-note'>"
        "This app visualizes your progressive multiple sequence alignment pipeline:<br><br>"
        "• pairwise distance matrix<br>"
        "• UPGMA guide tree<br>"
        "• final progressive alignment"
        "</div>",
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# Hero
# ------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🧬 Progressive MSA Visualizer</div>
    <div class="hero-subtitle">
        Paste protein sequences, compute pairwise distances, build the guide tree,
        and inspect the final progressive multiple sequence alignment — all in one place.
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Input section
# ------------------------------------------------------------
left, right = st.columns([1.7, 1])

with left:
    st.markdown("<div class='section-header'>Input Sequences</div>", unsafe_allow_html=True)
    sequence_text = st.text_area(
        "Paste one sequence per line",
        height=220,
        key="sequence_input",
        placeholder="Example:\nMKTAYIAKQRQISFVKSHFSRQ\nMKTAYIAKQRQISFVKSHFSRL\nMKTAYIAQRQISFVKSHFSRQ"
    )

with right:
    st.markdown("""
    <div class="soft-card">
        <div class="soft-card-title">What this app shows</div>
        <div class="soft-card-text">
            It takes your sequences, computes pairwise distances, builds a guide tree,
            and progressively merges profiles until one final alignment is produced.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <span class="pill">Profiles</span>
    <span class="pill">Distance Matrix</span>
    <span class="pill">UPGMA</span>
    <span class="pill">Progressive Alignment</span>
    """, unsafe_allow_html=True)

run_alignment = st.button("🚀 Run Progressive Alignment", use_container_width=True)


# ------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------
if run_alignment:
    sequences = parse_sequences(sequence_text)
    valid, msg = validate_sequences(sequences)

    if not valid:
        st.error(msg)
        st.stop()

    try:
        with st.spinner("Running the alignment pipeline..."):
            D = distance_matrix(sequences, gap=gap_penalty)
            Z = compute_linkage_matrix(D)
            dendro_fig = plot_dendrogram_from_Z(Z, len(sequences))
            heatmap_fig = plot_distance_heatmap(D)

            result = progressive_align(sequences, gap=gap_penalty)
            aligned_sequences = result.sequences

            alignment_df = build_alignment_table(aligned_sequences)
            Z_df = pd.DataFrame(
                Z,
                columns=["left_cluster", "right_cluster", "merge_distance", "cluster_size"]
            )

        # ----------------------------------------------------
        # Summary metrics
        # ----------------------------------------------------
        st.markdown("<div class='section-header'>Summary</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Input Sequences</div>
                <div class="metric-value">{len(sequences)}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Gap Penalty</div>
                <div class="metric-value">{gap_penalty}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Aligned Length</div>
                <div class="metric-value">{result.length}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Returned Sequences</div>
                <div class="metric-value">{result.n_seq}</div>
            </div>
            """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # Original sequences
        # ----------------------------------------------------
        st.markdown("<div class='section-header'>Original Sequences</div>", unsafe_allow_html=True)
        for i, seq in enumerate(sequences, start=1):
            st.markdown(
                f"""
                <div class="seq-card">
                    <span class="seq-label">S{i}</span>
                    <span class="seq-text">{seq}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Tabs
        # ----------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📏 Distance Matrix",
            "🌳 Guide Tree",
            "🧬 Final Alignment",
            "📋 Tables"
        ])

        with tab1:
            st.markdown("<div class='section-header'>Pairwise Distance Matrix</div>", unsafe_allow_html=True)
            st.dataframe(D.round(4), use_container_width=True)
            st.pyplot(heatmap_fig)

        with tab2:
            st.markdown("<div class='section-header'>Guide Tree</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='tiny-note'>"
                "Each row of the linkage matrix says: merge the left cluster and right cluster "
                "at this distance, producing a cluster of this size."
                "</div>",
                unsafe_allow_html=True
            )
            st.dataframe(Z_df.round(4), use_container_width=True)
            st.pyplot(dendro_fig)

        with tab3:
            st.markdown("<div class='section-header'>Final Progressive Alignment</div>", unsafe_allow_html=True)

            recovered = [seq.replace("-", "") for seq in aligned_sequences]
            all_same_len = len(set(len(seq) for seq in aligned_sequences)) == 1

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="tiny-note">
                        <b>All aligned sequences same length:</b> {all_same_len}<br>
                        <b>Recovered originals match input count:</b> {len(recovered) == len(sequences)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            for i, seq in enumerate(aligned_sequences, start=1):
                st.markdown(
                    f"""
                    <div class="seq-card">
                        <span class="seq-label">S{i}</span>
                        <span class="seq-text">{style_alignment(seq)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            fasta_text = build_fasta_output(aligned_sequences)
            st.download_button(
                "⬇️ Download Alignment as FASTA",
                data=fasta_text,
                file_name="progressive_alignment.fasta",
                mime="text/plain"
            )

        with tab4:
            st.markdown("<div class='section-header'>Alignment Table</div>", unsafe_allow_html=True)
            st.dataframe(alignment_df, use_container_width=True)

            st.markdown("<div class='section-header'>Recovered Original Sequences</div>", unsafe_allow_html=True)
            for i, seq in enumerate(recovered, start=1):
                st.markdown(
                    f"""
                    <div class="seq-card">
                        <span class="seq-label">S{i}</span>
                        <span class="seq-text">{seq}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.success("Progressive alignment completed successfully.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")