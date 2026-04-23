export type MethodGroup = "group1" | "group2";

export type Method = {
  id: string;
  name: string;
  group: MethodGroup;
};

export type GroupMethods = {
  group1: Method[];
  group2: Method[];
};

export const methods: GroupMethods = {
  group1: [
    { id: "LIME", name: "LIME", group: "group1" },
    { id: "SHAP", name: "SHAP", group: "group1" },
    { id: "IG", name: "IG", group: "group1" },
  ],
  group2: [
    { id: "Attention", name: "Attention", group: "group2" },
    { id: "Occlusion", name: "Occlusion", group: "group2" },
    { id: "Gradcam", name: "Grad CAM", group: "group2" },
  ],
};

const _check: GroupMethods = methods; 