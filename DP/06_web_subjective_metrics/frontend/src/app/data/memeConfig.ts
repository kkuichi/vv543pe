export interface MemeMeta {
    title: string;
    description: string;
    classification: string;
    // confidence: string;
  }
  
  export const memeConfig: Record<string, MemeMeta> = {
    group1: {
      title: "Text Model - BERT",
      description: `
        For this meme, we first generated a detailed textual description using a large language model (LLM). This description captures the visual elements and context of the meme in plain text form.

        The generated description was then processed by a text-based AI model to classify the meme.

        The explanation visualizations below show which words in the generated description had the strongest influence on the model’s final decision.
        More intense highlights indicate a greater impact on the prediction.
        `,
      classification: "Hateful",
    //   confidence: "87.3%",
    },
  
    group2: {
      title: "Multimodal Model - HateCLIPper",
      description: `
        For this meme, we used a multimodal model. 

        Instead of relying on text alone, this approach allows the system to capture interactions between visual elements and written content within the meme.

        The explanation visualizations below highlight which regions of the image contributed most strongly to the model’s prediction. 
        More intense areas indicate a greater influence on the final decision.
        `,
        classification: "Hateful",
            //   confidence: "91.2%",
            },
          };