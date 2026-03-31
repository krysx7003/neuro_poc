import type { Document } from "../../types/components";

interface Props {
    doc: Document;
}

function RODOCard({ doc }: Props) {
    return <h2>"I am a {doc.name} RODOCard"</h2>;
}

export default RODOCard;
