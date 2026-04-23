import { memeConfig } from "../data/memeConfig";
import memeImg  from "../data/group1_2.jpg";
import limeImg from "../data/bert_lime.jpg";
import shapImg from "../data/bert_shap_2.jpg";
import igImg from "../data/bert_ig.jpg";
import { useEffect, useState } from "react";
import { Header } from "../components/Header";
import { MemeDisplay } from "../components/MemeDisplay";
import { ExplanationCard } from "../components/ExplanationCard";
import { RankingSection } from "../components/RankingSection";
import { methods, type Method } from "../data/methods"; 


interface Group1Props {
  studentId: number;          
  // methodOrder: string[];     
  onComplete: (data: any) => void; 
}


export default function Group1({ studentId, onComplete }: Group1Props) {
  const [explanations, setExplanations] = useState<Record<string, Record<string, number>>>({});

  const groupMethods: Method[] = methods.group1;

  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  const memeUrl = memeImg;
    const methodImages: Record<string, string> = {
      LIME: limeImg,
      SHAP: shapImg,
      IG: igImg,
    };

  const handleExplanationRate = (methodId: string, criteriaRatings: Record<string, number>) => {
    console.log(`Rate for ${methodId}:`, criteriaRatings);
    setExplanations(prev => ({
      ...prev,
      [methodId]: criteriaRatings
    }));
  };


  const isMethodFullyRated = (methodId: string) => {
    const ratings = explanations[methodId];
    if (!ratings) return false;
    return Object.keys(ratings).length === 6;
  };

 
  const allMethodsRated = groupMethods.every(method => isMethodFullyRated(method.id));

  const handleRankingSubmit = (rankingData: any) => {
    if (!allMethodsRated) {
      alert("Please evaluate all methods");
      return;
    }

    const allData = {
      explanations: explanations,    
      rankings: rankingData, 
      completed_at: new Date().toISOString()
    };

    console.log("Submitting all data:", allData);
    onComplete(allData); 
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-8 py-24">
        <Header />
        
        <MemeDisplay
          imageUrl={memeUrl}
          {...memeConfig.group1}
        />

        <div className="mb-12">
          <h2 className="text-center mb-8">Evaluate Each Explanation Method</h2>
          <p className="text-center text-muted-foreground mb-12 max-w-3xl mx-auto">
            Rate each criterion using stars from 1 (lowest) to 5 (highest)
            <br /><br />
            <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full">
              Click on the image to see the full explanation
            </span>
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {groupMethods.map((method) => (
              <div key={method.id} className="relative">
                <ExplanationCard
                  method={method.id}
                  imageUrl={methodImages[method.id]}
                  onRate={(ratings) => handleExplanationRate(method.id, ratings)}
                />
                {!isMethodFullyRated(method.id) && (
                  <div className="absolute top-2 right-2 bg-yellow-500 text-black px-2 py-1 rounded text-xs">
                    ⚠️ Rate all 6 criteria
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <RankingSection
          methods={methods.group1}
          onSubmit={handleRankingSubmit}
          allRated={allMethodsRated}
        />

        <div className="mt-8 text-center">
          <p className="text-sm text-muted-foreground">
            {allMethodsRated 
              ? "✅ All methods evaluated! Now set the final ranking." 
              : `⚠️ ${groupMethods.filter(m => !isMethodFullyRated(m.id)).length} methods need complete evaluation`}
          </p>
        </div>
      </div>

      <footer className="border-t border-border mt-24">
        <div className="max-w-7xl mx-auto px-8 py-8">
          <p className="text-center text-sm text-muted-foreground">
            University Research Project • Diploma Thesis 2026 • Confidential
          </p>
        </div>
      </footer>
    </div>
  );
}











