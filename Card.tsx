import React from 'react';

interface CardProps {
  title: string;
  onClick: () => void;
}

const Card: React.FC<CardProps> = ({ title, onClick }) => {
  return (
    <div
      className="card card-hover cursor-pointer"
      onClick={onClick}
    >
      <h2 className="text-xl font-semibold">{title}</h2>
    </div>
  );
};

export default Card;