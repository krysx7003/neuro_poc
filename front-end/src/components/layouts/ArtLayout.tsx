import type { Document } from "../../types/components";

interface Props {
    doc: Document;
}

function ArtCard({ doc }: Props) {
    return <h2>"I am a {doc.name} ArtCard"</h2>;
}

export default ArtCard;
