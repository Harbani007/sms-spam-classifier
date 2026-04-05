import { useState } from "react";
import { Send, Shield, AlertTriangle, Loader2 } from "lucide-react";

// ---------------------------------------------------------------------------
// API configuration
// Set VITE_API_BASE_URL in your .env.local to point to the backend.
// Falls back to demo mode (keyword matching) if the env var is not set.
// ---------------------------------------------------------------------------
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;

interface ApiResult {
  predicted_label: "ham" | "spam";
  score: number | null;
  score_type: string;
  model_name: string;
}

interface ClassifyResult {
  text: string;
  label: "ham" | "spam";
  score: number | null;
  score_type: string;
  model_name: string;
  source: "api" | "demo";
}

const sampleMessages = [
  { text: "Congratulations! You have won a £1,000 prize. Call now!", label: "spam" as const },
  { text: "Hey, are we still on for lunch tomorrow?", label: "ham" as const },
  { text: "FREE entry in 2 a weekly competition to win FA Cup final tkts!", label: "spam" as const },
  { text: "Can you pick up some milk on the way home?", label: "ham" as const },
  { text: "URGENT! Your mobile No was awarded a £2,000 Bonus Prize", label: "spam" as const },
  { text: "I'll be there in 10 minutes, traffic is bad", label: "ham" as const },
];

// Keyword-based fallback for when the API is not configured
function _demoClassify(text: string): "ham" | "spam" {
  const spamKeywords = ["free", "win", "won", "prize", "congratulations", "urgent",
    "call now", "bonus", "£", "$", "click", "claim", "reward"];
  const lower = text.toLowerCase();
  const hits = spamKeywords.filter((kw) => lower.includes(kw)).length;
  const hasStrong = ["won", "prize", "bonus", "claim", "reward"].some((kw) => lower.includes(kw));
  return hits >= 2 || (hasStrong && lower.trim().split(/\s+/).length > 2) ? "spam" : "ham";
}

async function classifyViaApi(text: string): Promise<ApiResult> {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? `Server error ${res.status}`);
  }
  return res.json() as Promise<ApiResult>;
}

const InferenceDemo = () => {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const classify = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setResult(null);

    if (API_BASE_URL) {
      try {
        const api = await classifyViaApi(trimmed);
        setResult({
          text: trimmed,
          label: api.predicted_label,
          score: api.score,
          score_type: api.score_type,
          model_name: api.model_name,
          source: "api",
        });
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "API error — check that the backend is running.");
      }
    } else {
      // Demo mode: no backend required
      setResult({
        text: trimmed,
        label: _demoClassify(trimmed),
        score: null,
        score_type: "demo",
        model_name: "demo (keyword heuristic)",
        source: "demo",
      });
    }

    setLoading(false);
  };

  return (
    <section className="py-24 px-6 bg-secondary/20">
      <div className="container max-w-4xl">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold font-display text-gradient-accent mb-4">
            Interactive Demo
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            {API_BASE_URL
              ? "Connected to the FastAPI backend."
              : "Running in demo mode — set VITE_API_BASE_URL to connect to the real model."}
          </p>
        </div>

        {/* Input */}
        <div className="flex gap-3 mb-8">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && input.trim() && classify(input)}
            placeholder="Type an SMS message..."
            className="flex-1 px-5 py-3 rounded-lg border border-border bg-card text-foreground font-body placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
          <button
            onClick={() => input.trim() && classify(input)}
            disabled={loading || !input.trim()}
            className="px-6 py-3 rounded-lg bg-primary text-primary-foreground font-display font-semibold text-sm hover:opacity-90 disabled:opacity-50 transition-opacity flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            Classify
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 rounded-lg border border-destructive/30 bg-destructive/5 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className={`mb-8 p-5 rounded-xl border ${result.label === "spam" ? "border-destructive/30 bg-destructive/5" : "border-green-500/30 bg-green-500/5"} flex items-start gap-4`}>
            {result.label === "spam" ? (
              <AlertTriangle className="w-6 h-6 text-destructive flex-shrink-0 mt-0.5" />
            ) : (
              <Shield className="w-6 h-6 text-green-500 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1 min-w-0">
              <div className={`text-sm font-display font-bold uppercase tracking-wider ${result.label === "spam" ? "text-destructive" : "text-green-500"}`}>
                Prediction: {result.label.toUpperCase()}
              </div>
              <p className="text-sm text-muted-foreground mt-1 break-words">"{result.text}"</p>

              {result.score !== null && (
                <div className="text-xs text-muted-foreground/70 mt-2 space-y-1">
                  {result.score_type === "probability" ? (
                    <>
                      <p>
                        Confidence: {((result.label === "spam" ? result.score : 1 - result.score) * 100).toFixed(2)}% ({result.label.toUpperCase()})
                      </p>
                      <p>
                        Spam probability: {(result.score * 100).toFixed(2)}% · Ham probability: {((1 - result.score) * 100).toFixed(2)}%
                      </p>
                      <p>
                        Model: {result.model_name}
                      </p>
                    </>
                  ) : result.score_type === "decision_score" ? (
                    <>
                      <p>
                        Decision score: {result.score.toFixed(4)}
                      </p>
                      <p>
                        Model: {result.model_name}
                      </p>
                    </>
                  ) : (
                    <p>
                      Score: {result.score.toFixed(4)} ({result.score_type}) · Model: {result.model_name}
                    </p>
                  )}
                </div>
              )}

              {result.source === "demo" && (
                <p className="text-xs text-muted-foreground/60 mt-2">
                  Demo mode: keyword-based heuristic, not the trained backend model.
                </p>
              )}
            </div>
          </div>
        )}

        {/* Sample messages */}
        <div className="grid sm:grid-cols-2 gap-3">
          {sampleMessages.map((msg) => (
            <button
              key={msg.text}
              onClick={() => { setInput(msg.text); classify(msg.text); }}
              className="text-left p-4 rounded-lg border border-border bg-card/60 hover:border-primary/30 transition-colors group"
            >
              <div className="flex items-center gap-2 mb-1.5">
                {msg.label === "spam" ? (
                  <AlertTriangle className="w-3.5 h-3.5 text-destructive" />
                ) : (
                  <Shield className="w-3.5 h-3.5 text-green-500" />
                )}
                <span className={`text-xs font-display uppercase tracking-wider ${msg.label === "spam" ? "text-destructive" : "text-green-500"}`}>
                  {msg.label}
                </span>
              </div>
              <p className="text-sm text-muted-foreground group-hover:text-foreground transition-colors line-clamp-2">
                {msg.text}
              </p>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};

export default InferenceDemo;
