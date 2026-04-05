import { ArrowRight } from "lucide-react";

const steps = [
  {
    label: "Raw Text",
    detail: "SMS messages",
    color: "text-muted-foreground",
    borderColor: "border-muted-foreground/30",
  },
  {
    label: "TextPreprocessor",
    detail: "Regex → lowercase → stopwords → Porter stemming",
    color: "text-primary",
    borderColor: "border-primary/30",
  },
  {
    label: "Vectorizer",
    detail: "CountVectorizer (BoW) or TfidfVectorizer",
    color: "text-accent",
    borderColor: "border-accent/30",
  },
  {
    label: "Classifier",
    detail: "LR | LinearSVC | MultinomialNB",
    color: "text-success",
    borderColor: "border-success/30",
  },
  {
    label: "Prediction",
    detail: "ham / spam",
    color: "text-primary",
    borderColor: "border-primary/30",
  },
];

const PipelineSection = () => {
  return (
    <section id="pipeline" className="py-24 px-6">
      <div className="container">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold font-display text-gradient-primary mb-4">
            End-to-End Pipeline
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Fully sklearn-compatible pipeline — raw text in, prediction out. Pickle the whole thing for deployment.
          </p>
        </div>

        {/* Pipeline flow */}
        <div className="flex flex-col lg:flex-row items-center justify-center gap-2 lg:gap-0">
          {steps.map((step, i) => (
            <div key={step.label} className="flex items-center">
              <div className={`relative px-6 py-5 rounded-xl border ${step.borderColor} bg-card/60 backdrop-blur-sm min-w-[180px] text-center`}>
                <div className={`text-sm font-display font-bold ${step.color} tracking-wider`}>
                  {step.label}
                </div>
                <div className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                  {step.detail}
                </div>
              </div>
              {i < steps.length - 1 && (
                <ArrowRight className="w-5 h-5 text-muted-foreground mx-2 hidden lg:block flex-shrink-0" />
              )}
            </div>
          ))}
        </div>

        {/* Code snippet */}
        <div className="mt-16 max-w-3xl mx-auto">
          <div className="rounded-xl border border-border bg-card overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-secondary/30">
              <div className="w-3 h-3 rounded-full bg-destructive/60" />
              <div className="w-3 h-3 rounded-full bg-accent/60" />
              <div className="w-3 h-3 rounded-full bg-success/60" />
              <span className="ml-3 text-xs text-muted-foreground font-display">spam_detection.py</span>
            </div>
            <pre className="p-6 text-sm leading-relaxed overflow-x-auto">
              <code className="font-display">
                <span className="text-muted-foreground">{"# Fully end-to-end: raw text → prediction"}</span>{"\n"}
                <span className="text-accent">pipeline</span> = <span className="text-primary">Pipeline</span>([{"\n"}
                {"    "}(<span className="text-success">"pre"</span>, <span className="text-primary">TextPreprocessor</span>()),{"\n"}
                {"    "}(<span className="text-success">"vec"</span>, <span className="text-primary">TfidfVectorizer</span>(<span className="text-accent">max_features</span>=4000)),{"\n"}
                {"    "}(<span className="text-success">"clf"</span>, <span className="text-primary">LinearSVC</span>()),{"\n"}
                ]){"\n\n"}
                <span className="text-muted-foreground">{"# Train & predict — no external preprocessing needed"}</span>{"\n"}
                <span className="text-accent">pipeline</span>.fit(msgs_train, y_train){"\n"}
                <span className="text-accent">pred</span> = <span className="text-accent">pipeline</span>.predict([<span className="text-success">"You won a free iPhone!"</span>]){"\n"}
                <span className="text-muted-foreground">{"# → spam ✓"}</span>
              </code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PipelineSection;
