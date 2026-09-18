"""
SAS821S Lab 3 - Part C: Malware Propagation and Security-Control Simulation
Run top to bottom.  Needs: pandas, numpy, matplotlib, networkx
Data files must sit in a 'data' folder next to this script.
Outputs: simulation_summary.csv, sensitivity_analysis.csv,
         figure3_reach_probability.png, figure4_spread_size.png, figure5_spread_over_time.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
SEED = 8213026            # fixed seed (same as the lab starter file)
N_ITER = 2000             # Monte Carlo iterations per scenario (brief requires >= 1000)
T_MAX = 12                # simulation horizon in time steps (stopping condition)
PATIENT_ZERO = "VENDOR-LT-07"
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

print("Versions:", "python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| networkx", nx.__version__, "| matplotlib", matplotlib.__version__)

# ---------------------------------------------------------------- C1: load + build directed graph
assets = pd.read_csv(DATA / "egs_asset_inventory.csv")
posture = pd.read_csv(DATA / "egs_host_security_posture.csv")
edges = pd.read_csv(DATA / "egs_network_topology_edges.csv")
scenarios = pd.read_csv(DATA / "egs_simulation_scenarios.csv")

nodes = assets.merge(posture[["asset_id", "operating_system", "edr_present", "susceptibility"]], on="asset_id")
assert len(nodes) == len(assets) == 24, "asset/posture mismatch"
node_ix = {a: i for i, a in enumerate(nodes["asset_id"])}
N = len(nodes)

G = nx.from_pandas_edgelist(edges, "source_asset", "target_asset", edge_attr=True, create_using=nx.DiGraph)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges; start node = {PATIENT_ZERO}")
print("Assets reachable from start node (topology only):", len(nx.descendants(G, PATIENT_ZERO)))

zone = nodes["zone"].to_numpy()
is_ot = zone == "OT"
is_ot_dmz = zone == "OT-DMZ"
is_safety = zone == "Safety"
is_critical = (nodes["criticality"] == "Critical").to_numpy()
susc = nodes["susceptibility"].to_numpy(dtype=float)
# assets that an EDR-based rapid-isolation action can actually contain: Windows hosts with EDR
isolatable = (nodes["operating_system"].str.contains("Windows", na=False) & (nodes["edr_present"] == "Yes")).to_numpy()

src = edges["source_asset"].map(node_ix).to_numpy()
dst = edges["target_asset"].map(node_ix).to_numpy()
base_p = edges["base_transmission_probability"].to_numpy(dtype=float)
# "DMZ-to-OT and OT lateral paths": any edge that enters or moves within the OT side
ot_side = {"OT-DMZ", "OT", "Safety"}
seg_edge = (edges["target_zone"].isin(ot_side) & edges["source_zone"].isin({"DMZ"} | ot_side)).to_numpy()
# vendor access path patched in scenario S1 (vendor laptop, VPN gateway, vendor broker)
vendor_path = np.array([node_ix[a] for a in ["VENDOR-LT-07", "VPN-GW-01", "VENDOR-DMZ-01"]])
print(f"Edges treated as OT-side paths for segmentation: {seg_edge.sum()} of {len(edges)}")
print(f"Isolatable (Windows + EDR) assets: {isolatable.sum()} of {N}")


# ---------------------------------------------------------------- simulation engine
def run_scenario(sc, n_iter=N_ITER, seed=SEED, t_max=T_MAX, prob_scale=1.0):
    """
    Discrete-time independent-cascade model.
    - Each step, every infected, uncontained host attempts to infect each not-yet-infected
      out-neighbour once. P(success) = edge base probability x target susceptibility x control factors.
    - patch_factor lowers susceptibility of the vendor-path assets.
    - segmentation_factor lowers probability on OT-side edges.
    - After `isolation_delay_steps` steps of being infected, an isolatable (Windows+EDR) host's
      outgoing probability is multiplied by post_detection_transmission_factor.
    - Stops when no infected host can spread further or after t_max steps.
    """
    rng = np.random.default_rng(seed)
    s = susc.copy()
    s[vendor_path] = s[vendor_path] * sc["patch_factor"]
    p_edge = base_p * s[dst] * prob_scale
    p_edge = np.where(seg_edge, p_edge * sc["segmentation_factor"], p_edge)
    p_edge = np.clip(p_edge, 0, 1)
    delay, post = sc["isolation_delay_steps"], sc["post_detection_transmission_factor"]

    out = []
    curve = np.zeros((n_iter, t_max + 1))
    for it in range(n_iter):
        infected_at = np.full(N, -1)
        infected_at[node_ix[PATIENT_ZERO]] = 0
        ot_time = np.nan
        for t in range(1, t_max + 1):
            age = np.where(infected_at >= 0, (t - 1) - infected_at, -1)          # steps since infection
            contained = isolatable & (age >= delay)                                  # rapid-isolation kicks in
            active = (infected_at[src] >= 0) & (infected_at[dst] < 0)
            p = p_edge * np.where(contained[src], post, 1.0)
            hit = active & (rng.random(len(p)) < p)
            newly = np.unique(dst[hit])
            if len(newly):
                infected_at[newly] = t
                if np.isnan(ot_time) and is_ot[newly].any():
                    ot_time = t
            curve[it, t] = (infected_at >= 0).sum()
            if not active.any():                                                     # nothing left to infect
                curve[it, t:] = (infected_at >= 0).sum()
                break
        inf = infected_at >= 0
        out.append((inf[is_ot].any(), inf[is_ot_dmz].any(), inf[is_safety].any(),
                    (inf & is_critical).sum(), inf.sum(), ot_time))
    res = pd.DataFrame(out, columns=["reach_ot", "reach_ot_dmz", "reach_safety",
                                     "critical_inf", "total_inf", "time_to_ot"])
    return res, curve


# ---------------------------------------------------------------- C2: run every scenario
rows, curves = [], {}
for _, sc in scenarios.iterrows():
    res, curve = run_scenario(sc)
    curves[sc["scenario_id"]] = curve.mean(axis=0)
    rows.append({
        "scenario_id": sc["scenario_id"], "scenario_name": sc["scenario_name"], "iterations": N_ITER,
        "p_reach_ot": res["reach_ot"].mean(),
        "p_reach_ot_dmz": res["reach_ot_dmz"].mean(),
        "p_reach_safety_zone": res["reach_safety"].mean(),
        "mean_critical_assets_infected": res["critical_inf"].mean(),
        "mean_total_assets_infected": res["total_inf"].mean(),
        "median_time_to_ot_steps": res["time_to_ot"].median(),        # among runs that reached OT
        "n_runs_reaching_ot": int(res["reach_ot"].sum()),
        "p95_total_assets_infected": res["total_inf"].quantile(0.95),
    })
summary = pd.DataFrame(rows)

base_row = summary.loc[summary["scenario_id"] == "S0"].iloc[0]
for col, tag in [("p_reach_ot", "ot"), ("p_reach_safety_zone", "safety"),
                 ("mean_critical_assets_infected", "critical")]:
    summary[f"abs_reduction_{tag}"] = base_row[col] - summary[col]
    summary[f"rel_reduction_{tag}_pct"] = (base_row[col] - summary[col]) / base_row[col] * 100 if base_row[col] else np.nan

summary = summary.round(4)
summary.to_csv(BASE / "simulation_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n=== simulation_summary.csv ===")
print(summary.to_string(index=False))

# ---------------------------------------------------------------- figures
labels = summary["scenario_id"] + "\n" + summary["scenario_name"].str.replace(" ", "\n", n=1)
x = np.arange(len(summary)); w = 0.38

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w / 2, summary["p_reach_ot"] * 100, w, label="Reaches OT zone", color="#c0392b")
b2 = ax.bar(x + w / 2, summary["p_reach_safety_zone"] * 100, w, label="Reaches safety zone", color="#2c3e50")
ax.bar_label(b1, fmt="%.1f%%", fontsize=8); ax.bar_label(b2, fmt="%.1f%%", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Probability (% of Monte Carlo runs)")
ax.set_title(f"Figure 3: Probability malware from {PATIENT_ZERO} reaches OT / safety zone "
             f"({N_ITER} runs per scenario, {T_MAX} steps)")
ax.legend(); plt.tight_layout(); plt.savefig(BASE / "figure3_reach_probability.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w / 2, summary["mean_critical_assets_infected"], w, label="Mean critical assets infected", color="#e67e22")
b2 = ax.bar(x + w / 2, summary["mean_total_assets_infected"], w, label="Mean total assets infected", color="#7f8c8d")
ax.errorbar(x + w / 2, summary["mean_total_assets_infected"],
            yerr=[np.zeros(len(summary)), summary["p95_total_assets_infected"] - summary["mean_total_assets_infected"]],
            fmt="none", ecolor="black", capsize=4, label="Mean to 95th percentile")
ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Number of assets (out of 24)")
ax.set_title("Figure 4: Spread size by control scenario (mean and 95th percentile)")
ax.legend(); plt.tight_layout(); plt.savefig(BASE / "figure4_spread_size.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
for sid, c in curves.items():
    ax.plot(range(T_MAX + 1), c, marker="o", ms=3, label=sid)
ax.set_xlabel("Simulation time step"); ax.set_ylabel("Mean number of infected assets")
ax.set_title("Figure 5: Mean infected assets over time by scenario"); ax.legend(title="Scenario")
plt.tight_layout(); plt.savefig(BASE / "figure5_spread_over_time.png", dpi=150); plt.close()
print("\nSaved figure3_reach_probability.png, figure4_spread_size.png, figure5_spread_over_time.png")

# ---------------------------------------------------------------- C3: parameter sensitivity
sens = []
s0, s4 = (scenarios[scenarios["scenario_id"] == k].iloc[0] for k in ("S0", "S4"))
for name, sc in [("S0", s0), ("S4", s4)]:
    for label, kw in [("baseline", {}), ("edge prob x0.8", {"prob_scale": 0.8}), ("edge prob x1.2", {"prob_scale": 1.2}),
                      ("horizon 8 steps", {"t_max": 8}), ("horizon 20 steps", {"t_max": 20}),
                      ("different seed", {"seed": SEED + 1})]:
        r, _ = run_scenario(sc, **kw)
        sens.append({"scenario": name, "variation": label, "p_reach_ot": r["reach_ot"].mean(),
                     "p_reach_safety_zone": r["reach_safety"].mean(),
                     "mean_critical_assets_infected": r["critical_inf"].mean()})
sens = pd.DataFrame(sens).round(4)
sens.to_csv(BASE / "sensitivity_analysis.csv", index=False)
print("\n=== Sensitivity analysis (saved sensitivity_analysis.csv) ===")
print(sens.to_string(index=False))
