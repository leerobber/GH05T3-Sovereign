# backend/oss/lab/theory_lab_dashboard.py

import streamlit as st
import pandas as pd
import json
from backend.oss.meta_export import collect_meta_samples
from backend.oss.mvs import get_mvs
from backend.oss.omni_net import get_omni_net


def run_theory_lab_dashboard():
    mvs = get_mvs()
    substrate = mvs["substrate"]
    mind = mvs["mind"]
    economy = mvs["economy"]

    samples = collect_meta_samples(substrate, mind, economy)
    df = pd.DataFrame(samples)

    st.title("Elite Theory Lab — Evolution Dashboard (MVS + OmniWorlds)")

    st.header("Theorist population")
    theorists = df[df["is_theorist"]] if "is_theorist" in df.columns else df[df.get("role", "").str.contains("THEORIST", case=False, na=False)]
    st.metric("Number of Theorist genomes", len(theorists))
    st.metric("Total MVS genomes", len(df))

    if not theorists.empty:
        st.subheader("Theorist traits (means)")
        if isinstance(theorists.iloc[0]["traits"], dict):
            trait_means = {}
            for t in theorists.iloc[0]["traits"].keys():
                vals = [s["traits"].get(t, 0.55) for _, s in theorists.iterrows() if isinstance(s.get("traits"), dict)]
                trait_means[t] = round(sum(vals)/max(1,len(vals)), 3)
            st.json(trait_means)

    st.header("Recent Theory Lab activity (from memories)")
    theory_signals = []
    for _, row in theorists.iterrows():
        for m in (row.get("recent_memories") or []):
            if m.get("type") == "theory_lab" or m.get("theory_lab_cycle") is not None:
                theory_signals.append({
                    "genome": row["genome_id"],
                    "cycle": m.get("theory_lab_cycle"),
                    "world": m.get("world"),
                    "score": m.get("computed_score"),
                    "world_score": m.get("world_score"),
                    "canonical": m.get("canonical"),
                    "proposal": str(m.get("raw_proposal", ""))[:120] + "..."
                })
    if theory_signals:
        sig_df = pd.DataFrame(theory_signals)
        st.dataframe(sig_df)
        st.metric("Avg computed theory score (sampled)", round(sig_df["score"].mean(), 3))
    else:
        st.info("Run theory_lab.py to populate live theory_lab memory signals.")

    st.header("Worlds exercised")
    worlds = set()
    for _, row in df.iterrows():
        for m in (row.get("recent_memories") or []):
            if m.get("world"):
                worlds.add(m.get("world"))
    st.write(sorted(worlds) if worlds else "AlignmentWorld / MetaArchitectureWorld / VolatilityWorld (on run)")

    st.header("Omni-Net Beta (network layer)")
    try:
        net = get_omni_net()
        ns = net.stats()
        st.metric("Registered peers", ns.get("peer_count", 0))
        st.metric("Theories broadcast", ns.get("theories_published", 0))
        st.write("Top reputation peers:", ns.get("top_reputation", []))
        pulled = net.pull_canonical_memories(limit=3)
        if pulled:
            st.write("Recent net canonical samples:", len(pulled))
            for p in pulled[:1]:
                st.code(str(p)[:300])
    except Exception as e:
        st.info(f"Net not yet populated: {e}")

    st.header("Sample theorist raw proposal (for training data)")
    if not theorists.empty:
        for _, s in theorists.iterrows():
            mems = s.get("recent_memories", [])
            for m in mems:
                if m.get("raw_proposal"):
                    st.code(str(m.get("raw_proposal"))[:800])
                    break
            else:
                continue
            break

    st.header("Fitness history (sample theorist)")
    if not theorists.empty:
        sample = theorists.iloc[0]
        fh = sample.get("fitness_history", [])
        if fh:
            st.line_chart(pd.Series(fh))
        else:
            st.write("No fitness history yet (lab updates evolve traits).")

    st.header("Full meta export sample (last theorist)")
    if theorists is not None and len(theorists):
        st.json(theorists.iloc[-1].to_dict())


if __name__ == "__main__":
    run_theory_lab_dashboard()
