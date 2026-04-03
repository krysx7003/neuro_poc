import DocumentList from "./components/DocumentList";
import AnswerField from "./components/AnswerField";
import InputField from "./components/InputField";
import type { Document } from "./types/components";
import { useState } from "react";
import "./App.css";

function App() {
    const docs: Document[] = [
        { name: "Test", type: "rodo" },
        { name: "Test", type: "decision" },
        { name: "Test", type: "article" },
        { name: "Test", type: "nsa" },
        { name: "Test", type: "rodo" },
        { name: "Test", type: "article" },
        { name: "Test", type: "nsa" },
    ];
    const [answerVisible, setAnswerVisible] = useState(false);
    const [currentQuery, setCurrentQuery] = useState("");

    const isNotEmpty = (query: string) => {
        return query.length > 0 && query.trim().length > 0;
    };

    return (
        <>
            <div className={answerVisible ? "normal" : "centered"}>
                <InputField
                    onSearch={(query: string) => {
                        setCurrentQuery(query);
                        console.log("User query: " + query);
                        setAnswerVisible(isNotEmpty(query));
                    }}
                />
                {answerVisible ? (
                    <div>
                        <AnswerField text={currentQuery} />
                        <DocumentList full_docs={docs} />
                    </div>
                ) : (
                    <div />
                )}
            </div>
        </>
    );
}

export default App;
