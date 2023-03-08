import { CHAIN } from '@tonconnect/sdk';
import React, { useEffect, useRef, useState } from 'react';
import { useTonWallet } from 'src/hooks/useTonWallet';
import './style.scss';

const chainNames = {
	[CHAIN.MAINNET]: 'mainnet',
	[CHAIN.TESTNET]: 'testnet',
};

export function AppTitle() {
	const wallet = useTonWallet();
	const [clicks, setClicks] = useState(0);

	return (
		<>
			<div className="dapp-title" onClick={() => setClicks((x) => x + 1)}>
				<span className="dapp-title__text">Genlock</span>
				{wallet && <span className="dapp-title__badge">{chainNames[wallet.account.chain]}</span>}
			</div>
		</>
	);
}
