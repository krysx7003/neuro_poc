import type { Document, DocumentType } from "../types/components";
import Card from "./Card";
import ArtLayout from "./layouts/ArtLayout";
import DecisionLayout from "./layouts/DecisionLayout";
import NSALayout from "./layouts/NSALayout";
import RODOLayout from "./layouts/RODOLayout";
import "./DocumentList.css";
import { useState } from "react";

interface DocumentListProps {
    full_docs: Document[];
}
interface ItemProps {
    doc: Document;
}

const DocumentLayoutMap: Record<
    DocumentType,
    React.ComponentType<ItemProps>
> = {
    article: ArtLayout,
    decision: DecisionLayout,
    rodo: RODOLayout,
    nsa: NSALayout,
};

function DocumentList({ full_docs }: DocumentListProps) {
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [hoveredIndex, setHoveredIndex] = useState(-1);

    const decisions: Document[] = full_docs.filter(
        (doc) => doc.type === "decision",
    );
    const articles: Document[] = full_docs.filter(
        (doc) => doc.type === "article",
    );
    const nsa_docs: Document[] = full_docs.filter((doc) => doc.type === "nsa");
    const rodo_docs: Document[] = full_docs.filter(
        (doc) => doc.type === "rodo",
    );

    const headers: string[] = [
        `Wszystkie (${full_docs.length})`,
        `Artykuły UODO (${articles.length})`,
        `Decyzje UODO (${decisions.length})`,
        `Orzeczenia NSA (${nsa_docs.length})`,
        `RODO (${rodo_docs.length})`,
    ];

    const docs: Record<number, Document[]> = {
        0: full_docs,
        1: articles,
        2: decisions,
        3: nsa_docs,
        4: rodo_docs,
    };

    const getDocs = (index: number): Document[] => {
        return docs[index] ?? full_docs;
    };

    const selectedDocs: Document[] = getDocs(selectedIndex);
    return (
        <>
            <div className="row">
                {headers.map((item, index) => (
                    <div
                        className={
                            selectedIndex === index || hoveredIndex === index
                                ? "row-item active"
                                : "row-item"
                        }
                        onClick={() => {
                            setSelectedIndex(index);
                        }}
                        onMouseEnter={() => setHoveredIndex(index)}
                        onMouseLeave={() => setHoveredIndex(-1)}
                    >
                        <h3>{item}</h3>
                    </div>
                ))}
            </div>
            <ul className="document-display">
                {selectedDocs.length > 0 ? (
                    selectedDocs.map((doc) => {
                        const LayoutComponent = DocumentLayoutMap[doc.type];
                        return (
                            <li>
                                <Card key={doc.name}>
                                    <LayoutComponent doc={doc} />
                                </Card>
                            </li>
                        );
                    })
                ) : (
                    <h2>Nie znaleziono dokumentów</h2>
                )}
            </ul>
        </>
    );
}

export default DocumentList;
