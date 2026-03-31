import type { Document, DocumentType } from "../types/components";
import Card from "./Card";
import ArtLayout from "./layouts/ArtLayout";
import DecisionLayout from "./layouts/DecisionLayout";
import NSALayout from "./layouts/NSALayout";
import RODOLayout from "./layouts/RODOLayout";

interface DocumentListProps {
    docs: Document[];
}
interface ItemProps {
    doc: Document;
}

const DocumentMap: Record<DocumentType, React.ComponentType<ItemProps>> = {
    article: ArtLayout,
    decision: DecisionLayout,
    rodo: RODOLayout,
    nsa: NSALayout,
};

function DocumentList({ docs }: DocumentListProps) {
    return (
        <>
            <ul>
                {docs.map((doc) => {
                    const LayoutComponent = DocumentMap[doc.type];
                    return (
                        <li>
                            <Card key={doc.name}>
                                <LayoutComponent doc={doc} />
                            </Card>
                        </li>
                    );
                })}
            </ul>
        </>
    );
}

export default DocumentList;
