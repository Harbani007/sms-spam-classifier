import HeroSection from "@/components/HeroSection";
import PipelineSection from "@/components/PipelineSection";
import ModelComparison from "@/components/ModelComparison";
import FeaturesSection from "@/components/FeaturesSection";
import InferenceDemo from "@/components/InteractiveDemo";
import TechStack from "@/components/TechStack";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <HeroSection />
      <PipelineSection />
      <FeaturesSection />
      <ModelComparison />
      <InferenceDemo />
      <TechStack />
      <Footer />
    </div>
  );
};

export default Index;
