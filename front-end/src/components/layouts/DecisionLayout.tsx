import type { Document } from "../../types/components";

interface Props {
    doc: Document;
}

function DecisionCard({ doc }: Props) {
    return <h2>"I am a {doc.name} DecisionCard"</h2>;
}

export default DecisionCard;
