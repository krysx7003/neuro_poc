import DocumentList from "./components/DocumentList";
import AnswerField from "./components/AnswerField";
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
            <AnswerField
                text="
 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed condimentum placerat mauris, a posuere magna pharetra ac. Sed eros ipsum, ornare ac maximus in, rutrum eu augue. Maecenas tincidunt a nibh ac ullamcorper. Duis eget convallis sapien. Morbi sed tempor dui. In at finibus ante. Fusce et risus urna. Proin dignissim eros eu felis gravida, sed suscipit eros aliquet. Sed vitae augue tellus. In magna magna, consequat eu odio vitae, iaculis porttitor arcu. Curabitur euismod molestie tincidunt. Donec fringilla, mi a placerat consequat, urna ex egestas tellus, ac imperdiet felis est nec ante. Nullam a magna dictum est consectetur porttitor. Aliquam id dui at lectus varius semper eu vel mi. Quisque vehicula euismod leo, at faucibus ipsum.
            "
            />
            <DocumentList full_docs={docs} />
        </>
    );
}

export default App;
