import { Layers, FlaskConical, ShieldCheck, Package, BarChart3, Bug } from "lucide-react";

const features = [
  {
    icon: Layers,
    title: "Sklearn Pipeline Architecture",
    description: "Custom TextPreprocessor transformer baked into every pipeline — pickle the .pkl and deploy raw-text inference instantly.",
    color: "text-primary",
  },
  {
    icon: FlaskConical,
    title: "Rigorous Evaluation",
    description: "Stratified 5-fold cross-validation on both BoW & TF-IDF, plus identical hold-out split for fair comparison across 6 pipelines.",
    color: "text-accent",
  },
  {
    icon: ShieldCheck,
    title: "Reproducible Science",
    description: "Fixed random state, stratified splits, labeled constants, and deterministic artifact naming ensure every run is reproducible.",
    color: "text-success",
  },
  {
    icon: Package,
    title: "Deployment-Ready Artifacts",
    description: "Best pipeline auto-selected by F1 score, saved as a single .pkl file with embedded preprocessing — no external dependencies at inference.",
    color: "text-primary",
  },
  {
    icon: BarChart3,
    title: "Rich Visualizations",
    description: "Label distribution, confusion matrices, grouped bar charts, and BoW vs TF-IDF comparisons — all auto-saved as PNGs.",
    color: "text-accent",
  },
  {
    icon: Bug,
    title: "Error Analysis",
    description: "Automatic false-positive and false-negative breakdown with example messages — understand where your model fails and why.",
    color: "text-success",
  },
];

const FeaturesSection = () => {
  return (
    <section className="py-24 px-6">
      <div className="container">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold font-display text-gradient-primary mb-4">
            Why This Project Stands Out
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            A compact ML project focused on text classification, model comparison, and clear technical presentation.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f) => (
            <div
              key={f.title}
              className="group rounded-xl border border-border bg-card/60 p-6 hover:border-primary/30 hover:glow-primary transition-all duration-300"
            >
              <f.icon className={`w-8 h-8 ${f.color} mb-4 group-hover:scale-110 transition-transform`} />
              <h3 className="text-lg font-display font-bold text-foreground mb-2">{f.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
