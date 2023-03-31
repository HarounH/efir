import React from "react";

function MnistImage({url}) {
  const imgStyle = {
    display: 'block',
    margin: 'auto',
    width: '25%',
    height: '25%',
    objectFit: "cover",
  };

  const containerStyle = {
    backgroundColor: 'black',
    width: '100%',
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  };

  return (
    <div style={containerStyle}>
      <img src={url} style={imgStyle} alt="generated" />
    </div>
  );
}

export default MnistImage;
