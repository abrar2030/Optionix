import React, { useState } from "react";
import { Contract, ethers } from "ethers";

export default function TradingInterface() {
  const [size, setSize] = useState("");
  const [isLong, setIsLong] = useState(true);

  const executeTrade = async () => {
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const signer = provider.getSigner();
    const contract = new Contract("0x...", Futures_ABI, signer);
    await contract.openPosition(ethers.utils.parseEther(size), isLong, {
      value: ethers.utils.parseEther((size * 0.1).toString()),
    });
  };

  return (
    <div>
      <input type="number" onChange={(e) => setSize(e.target.value)} />
      <button onClick={() => setIsLong(!isLong)}>
        {isLong ? "Long" : "Short"}
      </button>
      <button onClick={executeTrade}>Execute Trade</button>
    </div>
  );
}
