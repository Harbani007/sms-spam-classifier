import { Github, Linkedin, Mail } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 px-6 border-t border-border">
      <div className="container flex flex-col md:flex-row items-center justify-between gap-6">
        <div>
          <div className="text-sm font-display text-foreground font-bold">SMS Spam Detection</div>
          <div className="text-xs text-muted-foreground mt-1">
            Built to showcase NLP classification, evaluation, and interactive model presentation.
          </div>
        </div>

        <div className="flex items-center gap-4">
          <a
            href="https://github.com/Harbani007"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-primary transition-colors"
          >
            <Github className="w-5 h-5" />
          </a>

          <a
            href="https://www.linkedin.com/in/harbanikaur1"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-primary transition-colors"
          >
            <Linkedin className="w-5 h-5" />
          </a>

          <a
            href="mailto:harbani.k1@gmail.com"
            className="text-muted-foreground hover:text-primary transition-colors"
          >
            <Mail className="w-5 h-5" />
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;