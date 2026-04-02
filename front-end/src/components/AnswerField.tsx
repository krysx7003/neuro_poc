import "./AnswerField.css";

interface Props {
    text: string;
}

function AnswerField({ text }: Props) {
    return (
        <>
            <div className="header">
                <div className="header-text">
                    <h3>Odpowiedź wygenerowana przez AI</h3>
                </div>
            </div>
            <div className="body">{text}</div>
        </>
    );
}

export default AnswerField;
