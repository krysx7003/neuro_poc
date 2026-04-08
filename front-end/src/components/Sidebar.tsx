import { useState } from "react";
import "./Sidebar.css";

interface Props {
    providers: string[];
}

const getProviderModels = (provider: string) => {
    return ["Volvo", "Saab", "Mercedes", "Audi"];
};
function Sidebar({ providers }: Props) {
    const [selectedProvider, setSelectedProvider] = useState(providers[0]);
    const models = getProviderModels(selectedProvider);

    return (
        <div className="sidebar">
            <h2>Opcje AI</h2>
            <h4>Provider</h4>
            <select onChange={(e) => setSelectedProvider(e.target.value)}>
                {providers.map((provider) => {
                    return <option value={provider}>{provider}</option>;
                })}
            </select>
            <h4>Model</h4>
            <select>
                {models.map((model) => {
                    return <option value={model}>{model}</option>;
                })}
            </select>
            <hr />
            <h2>Filtry wyszukiwania</h2>
            <h4>Data wydania dokumentu</h4>
            <h4>Rodzaj dokumentu</h4>
            <hr />
            <h2>Baza danych</h2>
            <h4>Decyzje UODO</h4>
            <h4>Artykuły UODO</h4>
            <h4>Orzeczenia NSA</h4>
            <h4>RODO</h4>
        </div>
    );
}

export default Sidebar;
