import type { Document } from "../../types/components";

interface Props {
    doc: Document;
}

function NSACard({ doc }: Props) {
    return <h2>"I am a {doc.name} NSACard"</h2>;
}

export default NSACard;
