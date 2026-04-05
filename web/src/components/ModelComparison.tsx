import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from "recharts";

const holdoutData = [
  { name: "LR (BoW)", Accuracy: 0.9839, F1: 0.9419, Precision: 0.9722, Recall: 0.9133 },
  { name: "LR (TF-IDF)", Accuracy: 0.9668, F1: 0.8750, Precision: 1.0, Recall: 0.7778 },
  { name: "SVC (BoW)", Accuracy: 0.9874, F1: 0.9565, Precision: 0.9706, Recall: 0.9429 },
  { name: "SVC (TF-IDF)", Accuracy: 0.9784, F1: 0.9205, Precision: 1.0, Recall: 0.8527 },
  { name: "MNB (BoW)", Accuracy: 0.9857, F1: 0.9474, Precision: 0.9855, Recall: 0.9121 },
  { name: "MNB (TF-IDF)", Accuracy: 0.9713, F1: 0.9007, Precision: 0.9556, Recall: 0.8514 },
];

const radarData = [
  { metric: "Accuracy", "LinearSVC (BoW)": 98.7, "LR (BoW)": 98.4, "MNB (BoW)": 98.6 },
  { metric: "F1", "LinearSVC (BoW)": 95.7, "LR (BoW)": 94.2, "MNB (BoW)": 94.7 },
  { metric: "Precision", "LinearSVC (BoW)": 97.1, "LR (BoW)": 97.2, "MNB (BoW)": 98.6 },
  { metric: "Recall", "LinearSVC (BoW)": 94.3, "LR (BoW)": 91.3, "MNB (BoW)": 91.2 },
];

const ModelComparison = () => {
  return (
    <section className="py-24 px-6 bg-secondary/20">
      <div className="container">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold font-display text-gradient-accent mb-4">
            Model Comparison
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            6 pipelines evaluated on identical stratified hold-out split. Best pipeline selected by F1 score.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Bar chart */}
          <div className="rounded-xl border border-border bg-card p-6">
            <h3 className="text-sm font-display text-primary tracking-wider uppercase mb-6">Hold-Out Metrics</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={holdoutData} margin={{ top: 5, right: 20, bottom: 50, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 14% 14%)" />
                <XAxis dataKey="name" tick={{ fill: "hsl(215 12% 50%)", fontSize: 10 }} angle={-25} textAnchor="end" />
                <YAxis domain={[0.75, 1.0]} tick={{ fill: "hsl(215 12% 50%)", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: "hsl(220 18% 7%)",
                    border: "1px solid hsl(220 14% 14%)",
                    borderRadius: "8px",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 12,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }} />
                <Bar dataKey="Accuracy" fill="hsl(174 72% 56%)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="F1" fill="hsl(45 96% 64%)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Precision" fill="hsl(142 71% 45%)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall" fill="hsl(262 83% 68%)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Radar chart */}
          <div className="rounded-xl border border-border bg-card p-6">
            <h3 className="text-sm font-display text-accent tracking-wider uppercase mb-6">BoW Models — Radar View</h3>
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="hsl(220 14% 14%)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "hsl(215 12% 50%)", fontSize: 11 }} />
                <PolarRadiusAxis domain={[88, 100]} tick={{ fill: "hsl(215 12% 50%)", fontSize: 10 }} />
                <Radar name="LinearSVC" dataKey="LinearSVC (BoW)" stroke="hsl(174 72% 56%)" fill="hsl(174 72% 56%)" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="LR" dataKey="LR (BoW)" stroke="hsl(45 96% 64%)" fill="hsl(45 96% 64%)" fillOpacity={0.1} strokeWidth={2} />
                <Radar name="MNB" dataKey="MNB (BoW)" stroke="hsl(142 71% 45%)" fill="hsl(142 71% 45%)" fillOpacity={0.1} strokeWidth={2} />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Results table */}
        <div className="mt-12 rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-6 py-4 border-b border-border">
            <h3 className="text-sm font-display text-primary tracking-wider uppercase">Detailed Results — Sorted by F1</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm font-display">
              <thead>
                <tr className="border-b border-border bg-secondary/30">
                  <th className="px-6 py-3 text-left text-muted-foreground font-medium">Model</th>
                  <th className="px-6 py-3 text-left text-muted-foreground font-medium">Feature</th>
                  <th className="px-6 py-3 text-right text-muted-foreground font-medium">Accuracy</th>
                  <th className="px-6 py-3 text-right text-muted-foreground font-medium">F1</th>
                  <th className="px-6 py-3 text-right text-muted-foreground font-medium">Precision</th>
                  <th className="px-6 py-3 text-right text-muted-foreground font-medium">Recall</th>
                </tr>
              </thead>
              <tbody>
                {holdoutData
                  .sort((a, b) => b.F1 - a.F1)
                  .map((row, i) => (
                    <tr key={row.name} className={`border-b border-border/50 ${i === 0 ? "bg-primary/5" : "hover:bg-secondary/20"} transition-colors`}>
                      <td className="px-6 py-3 text-foreground font-medium">{row.name.split(" (")[0]}</td>
                      <td className="px-6 py-3 text-muted-foreground">{row.name.includes("TF-IDF") ? "TF-IDF" : "BoW"}</td>
                      <td className="px-6 py-3 text-right text-foreground">{row.Accuracy.toFixed(4)}</td>
                      <td className={`px-6 py-3 text-right font-bold ${i === 0 ? "text-primary" : "text-foreground"}`}>{row.F1.toFixed(4)}</td>
                      <td className="px-6 py-3 text-right text-foreground">{row.Precision.toFixed(4)}</td>
                      <td className="px-6 py-3 text-right text-foreground">{row.Recall.toFixed(4)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelComparison;
