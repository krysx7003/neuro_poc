import DocumentList from "./components/DocumentList";
import type { Document } from "./types/components";

function App() {
    const docs: Document[] = [
        { name: "Test", type: "rodo" },
        // { name: "Test", type: "decision" },
        // { name: "Test", type: "article" },
        // { name: "Test", type: "nsa" },
    ];
    return (
        <>
            <DocumentList full_docs={docs} />
        </>
    );
}

export default App;
