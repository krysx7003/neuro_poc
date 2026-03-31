export type DocumentType = "article" | "decision" | "nsa" | "rodo";

export interface Document {
    name: string;
    type: DocumentType;
}
