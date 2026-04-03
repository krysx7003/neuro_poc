import { useState } from "react";
import "./InputField.css";

interface Props {
    onSearch: (query: string) => void;
}

function InputField({ onSearch }: Props) {
    const [query, setQuery] = useState("");
    return (
        <div className="query">
            <input
                className="query-input"
                placeholder="Wpisz treść pytania,sygnaturę lub temat"
                onChange={(e) => setQuery(e.target.value)}
            />
            <button
                className="search-btn"
                type="button"
                onClick={() => onSearch(query)}
            >
                <h3>Szukaj</h3>
            </button>
        </div>
    );
}

export default InputField;
