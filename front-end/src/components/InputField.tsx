import { useState } from "react";
import "./InputField.css";

function InputField() {
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
                onClick={() => console.log(`Test ${query}`)}
            >
                <h3>Szukaj</h3>
            </button>
        </div>
    );
}

export default InputField;
