import type { ReactNode } from "react";

interface Props {
    children: ReactNode;
}

function Card({ children }: Props) {
    return <>{children}</>;
}

export default Card;
