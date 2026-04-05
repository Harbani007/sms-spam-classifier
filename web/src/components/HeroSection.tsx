import { Github, FileText, Zap } from "lucide-react";

const metrics = [
  { label: "F1 Score", value: "0.97+", icon: Zap },
  { label: "Models Tested", value: "3 × 2", icon: FileText },
  { label: "Workflow", value: "NLP Pipeline", icon: Github },
];

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center hero-gradient overflow-hidden">

      <div className="container relative z-10 px-6 py-20 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary/20 bg-primary/5 mb-8">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse-glow" />
          <span className="text-sm font-display text-primary tracking-wider uppercase">
            Machine Learning Project
          </span>
        </div>

        <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold font-display leading-tight mb-6">
          <span className="text-gradient-primary">SMS Spam</span>
          <br />
          <span className="text-foreground">Detection</span>
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-12 font-body leading-relaxed">
          An NLP project showcase comparing <span className="text-primary">Logistic Regression</span>,{" "}
          <span className="text-accent">LinearSVC</span>, and{" "}
          <span className="text-success">Multinomial Naive Bayes</span>{" "}
          across <span className="text-foreground">Bag of Words</span> and{" "}
          <span className="text-foreground">TF-IDF</span> features, with a clean interface for model presentation and interactive demo testing.
        </p>

        <div className="flex flex-wrap justify-center gap-6 mb-12">
          {metrics.map((m) => (
            <div
              key={m.label}
              className="flex items-center gap-3 px-6 py-4 rounded-xl border border-border bg-card/50 backdrop-blur-sm glow-primary"
            >
              <m.icon className="w-5 h-5 text-primary" />
              <div className="text-left">
                <div className="text-2xl font-bold font-display text-foreground">
                  {m.value}
                </div>
                <div className="text-xs text-muted-foreground uppercase tracking-wider">
                  {m.label}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap justify-center gap-4">
          <a
            href="https://github.com/Harbani007/sms-spam-detection"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-8 py-3 rounded-lg bg-primary text-primary-foreground font-display font-semibold text-sm tracking-wider hover:opacity-90 transition-opacity"
          >
            <Github className="w-4 h-4" />
            View on GitHub
          </a>

          <a
            href="#pipeline"
            className="inline-flex items-center gap-2 px-8 py-3 rounded-lg border border-border text-foreground font-display font-semibold text-sm tracking-wider hover:bg-secondary transition-colors"
          >
            Explore Pipeline
          </a>
        </div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  );
};

export default HeroSection;