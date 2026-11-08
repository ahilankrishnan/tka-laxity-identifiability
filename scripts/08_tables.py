"""LaTeX table rows generated from the model definition (manuscript/tab_bundles.tex)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.model import BUNDLES, Ry, T_REF   # noqa: E402

GROUP = {"MCL": "MCL", "LCL": "LCL", "PCL": "PCL", "CAP": "Capsule"}
NAME = {"aMCL": "anterior", "iMCL": "intermediate", "pMCL": "posterior", "LCL": "single",
        "aPCL": "anterolateral", "pPCL": "posteromedial", "mCAP": "posteromedial",
        "lCAP": "posterolateral", "alCAP": "anterolateral"}


def main() -> None:
    """Write manuscript/tab_bundles.tex from the bundle definitions."""
    rows = []
    for n, (g, pf, pt) in BUNDLES.items():
        L = np.linalg.norm(np.array(pf) - (np.array(pt) + T_REF))
        e90 = np.linalg.norm(np.array(pf) - Ry(np.radians(90)) @ (np.array(pt) + T_REF)) / L - 1
        f = lambda v: "(" + ", ".join(f"${x:g}$" for x in v) + ")"
        rows.append(f"{GROUP[g]} & {NAME[n]} & {f(pf)} & {f(pt)} & {L:.0f} & ${100 * e90:+.0f}$\\\\")
    tpl = (ROOT / "manuscript" / "tab_bundles_template.txt").read_text(encoding="utf-8")
    tpl = tpl.replace(r"\input{tab_bundles_rows}", "\n".join(rows))
    (ROOT / "manuscript" / "tab_bundles.tex").write_text(tpl + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
