const techs = [
  { name: "Python", category: "Language" },
  { name: "scikit-learn", category: "ML Framework" },
  { name: "pandas", category: "Data" },
  { name: "NumPy", category: "Numerical" },
  { name: "NLTK", category: "NLP" },
  { name: "matplotlib", category: "Visualization" },
  { name: "seaborn", category: "Visualization" },
  { name: "pickle", category: "Serialization" },
];

const TechStack = () => {
  return (
    <section className="py-24 px-6">
      <div className="container">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold font-display text-gradient-primary mb-4">
            Tech Stack
          </h2>
        </div>

        <div className="flex flex-wrap justify-center gap-4 max-w-3xl mx-auto">
          {techs.map((t) => (
            <div
              key={t.name}
              className="px-6 py-3 rounded-lg border border-border bg-card/60 hover:border-primary/30 hover:glow-primary transition-all duration-300"
            >
              <div className="text-sm font-display font-bold text-foreground">{t.name}</div>
              <div className="text-xs text-muted-foreground">{t.category}</div>
            </div>
          ))}
        </div>

        {/* Quick start */}
        <div className="mt-16 max-w-2xl mx-auto">
          <div className="rounded-xl border border-border bg-card overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-secondary/30">
              <div className="w-3 h-3 rounded-full bg-destructive/60" />
              <div className="w-3 h-3 rounded-full bg-accent/60" />
              <div className="w-3 h-3 rounded-full bg-success/60" />
              <span className="ml-3 text-xs text-muted-foreground font-display">terminal</span>
            </div>
            <pre className="p-6 text-sm leading-loose overflow-x-auto">
              <code className="font-display">
                <span className="text-muted-foreground">$ </span><span className="text-primary">git clone</span> https://github.com/Harbani007/sms-spam-detection.git{"\n"}
                <span className="text-muted-foreground">$ </span><span className="text-primary">cd</span> sms-spam-detection{"\n"}
                <span className="text-muted-foreground">$ </span><span className="text-primary">pip install</span> -r requirements.txt{"\n"}
                <span className="text-muted-foreground">$ </span><span className="text-primary">python</span> spam_detection.py{"\n"}
                <span className="text-success">✓ Best pipeline: LinearSVC (BoW) — F1=0.9565</span>
              </code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TechStack;
