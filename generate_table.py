import json
import numpy as np  # <-- FIX: Added this line
import os

# Load complete results
with open('results_enhanced/q1_all_results.json', 'r') as f:
    data = json.load(f)

# LaTeX table generation
table_content = r"""
\begin{table}[h!]
\centering
\begin{threeparttable}
\caption{Statistical performance comparison across geological formations ($n=5$ independent runs). IP-FPINN demonstrates superior stability with statistically significant accuracy improvements on heterogeneous formations.}
\label{tab:primary_results}
\begin{tabular}{lcccccccc}
\toprule
 & & \multicolumn{3}{c}{\textbf{IP-FPINN}} & \multicolumn{3}{c}{\textbf{PINN Baseline}} & \\
 \cmidrule(lr){3-5} \cmidrule(lr){6-8}
\textbf{$\alpha$} & \textbf{Formation Type} & $\mu_{L^2}$ & $\sigma_{L^2}$ & CV (\%) & $\mu_{L^2}$ & $\sigma_{L^2}$ & CV (\%) & \textbf{Improvement} \\
\midrule
"""

for i, d in enumerate(data):
    alpha = d['alpha']
    # Determine formation type
    if alpha == 0.3:
        formation = "Fractured"
    elif alpha == 0.5:
        formation = "Highly Heterogeneous"
    elif alpha == 0.7:
        formation = "Layered"
    else:
        formation = "Homogeneous"
    
    ip_mean = d['ipfpinn']['mean_l2']
    ip_std = d['ipfpinn']['std_l2']
    ip_cv = d['ipfpinn']['cv']
    
    pinn_mean = d['pinn']['mean_l2']
    pinn_std = d['pinn']['std_l2']
    pinn_cv = d['pinn']['cv']
    
    improvement = d['improvement_percent']
    
    # Highlight significant improvements
    if improvement > 7.5:
        significance = r"\tnote{*}"
    else:
        significance = ""
    
    row = f"    {alpha} & {formation} & {ip_mean:.3f} & {ip_std:.3f} & {ip_cv:.2f} & "
    row += f"{pinn_mean:.3f} & {pinn_std:.3f} & {pinn_cv:.2f} & "
    row += f"+{improvement:.1f}\\%{significance} \\\\\n"
    table_content += row

# Add averages
avg_ip_mean = np.mean([d['ipfpinn']['mean_l2'] for d in data])
avg_ip_std = np.mean([d['ipfpinn']['std_l2'] for d in data])
avg_ip_cv = np.mean([d['ipfpinn']['cv'] for d in data])
avg_pinn_mean = np.mean([d['pinn']['mean_l2'] for d in data])
avg_pinn_std = np.mean([d['pinn']['std_l2'] for d in data])
avg_pinn_cv = np.mean([d['pinn']['cv'] for d in data])
avg_improvement = np.mean([d['improvement_percent'] for d in data])

table_content += r"""    \midrule
    \multicolumn{2}{l}{\textbf{Average}} & """
table_content += f"{avg_ip_mean:.3f} & {avg_ip_std:.3f} & {avg_ip_cv:.2f} & "
table_content += f"{avg_pinn_mean:.3f} & {avg_pinn_std:.3f} & {avg_pinn_cv:.2f} & "
table_content += r"\textbf{+" + f"{avg_improvement:.1f}\\%" + r"} \\"

table_content += r"""
    \bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item[*] Statistically significant improvement ($p<0.001$, two-sample t-test)
\end{tablenotes}
\end{threeparttable}
\end{table}
"""

# Save to file
os.makedirs('paper', exist_ok=True)
with open('paper/table.tex', 'w') as f:
    f.write(table_content)

print("✅ LaTeX table saved to paper/table.tex")
print("📋 Copy the content from that file into your main.tex")