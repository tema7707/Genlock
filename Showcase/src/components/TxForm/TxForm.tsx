import { SendTransactionRequest, CHAIN } from '@tonconnect/sdk';
import { Button, Input, Modal, notification, Typography } from 'antd';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import QRCode from 'react-qr-code';
import { Unity, useUnityContext } from 'react-unity-webgl';
import { useRecoilValueLoadable } from 'recoil';
import { addReturnStrategy, connector, sendTransaction } from 'src/connector';
import { useSlicedAddress } from 'src/hooks/useSlicedAddress';
import { useTonWallet } from 'src/hooks/useTonWallet';
import { useTonWalletConnectionError } from 'src/hooks/useTonWalletConnectionError';
import { walletsListQuery } from 'src/state/wallets-list';
import { isDesktop, isMobile, openLink } from 'src/utils';
import './style.scss';

const { Title, Text } = Typography;

export function TxForm() {
	const [tx, setTx] = useState<SendTransactionRequest | null>(null);
	const wallet = useTonWallet();
	const walletsList = useRecoilValueLoadable(walletsListQuery);

	useEffect(() => {
		if (wallet) {
			if (wallet.account.chain !== CHAIN.TESTNET) {
				notification.error({
					message: 'Mainnet is not available!',
					description: 'Please use your testnet wallet.',
				});
				connector.disconnect();
			} else {
				document.getElementById('loading')!.style.display = 'inline';
				document.getElementById('unity')!.style.display = 'inline';
			}
		}
	}, [wallet]);

	const onChange = useCallback(
		(value: object) => setTx((value as { updated_src: SendTransactionRequest }).updated_src),
		[],
	);

	const [modalUniversalLink, setModalUniversalLink] = useState('');
	const onConnectErrorCallback = useCallback(() => {
		setModalUniversalLink('');
		notification.error({
			message: 'Connection was rejected',
			description: 'Please approve connection to the dApp in your wallet.',
		});
	}, []);
	useTonWalletConnectionError(onConnectErrorCallback);

	const address = useSlicedAddress(wallet?.account.address, wallet?.account.chain);

	useEffect(() => {
		if (modalUniversalLink && wallet) {
			setModalUniversalLink('');
		}
	}, [modalUniversalLink, wallet]);

	const handleButtonClick = useCallback(async () => {
		// Use loading screen/UI instead (while wallets list is loading)
		if (!(walletsList.state === 'hasValue')) {
			setTimeout(handleButtonClick, 200);
		}

		if (!isDesktop() && walletsList.contents.embeddedWallet) {
			connector.connect({ jsBridgeKey: walletsList.contents.embeddedWallet.jsBridgeKey });
			return;
		}

		const tonkeeperConnectionSource = {
			universalLink: walletsList.contents.walletsList[0].universalLink,
			bridgeUrl: walletsList.contents.walletsList[0].bridgeUrl,
		};

		const universalLink = connector.connect(tonkeeperConnectionSource);

		if (isMobile()) {
			openLink(addReturnStrategy(universalLink, 'none'), '_blank');
		} else {
			setModalUniversalLink(universalLink);
		}
	}, [walletsList]);

	// Unity
	const { unityProvider, sendMessage, addEventListener, removeEventListener } = useUnityContext({
		loaderUrl: '/demo-dapp/build/webview.loader.js',
		dataUrl: '/demo-dapp/build/webview.data',
		frameworkUrl: '/demo-dapp/build/webview.framework.js',
		codeUrl: '/demo-dapp/build/webview.wasm',
	});

	const handleMint = useCallback(() => {
		document.getElementById('button_approve')!.style.display = 'inline';
	}, []);

	const handleLoad = useCallback(() => {
		document.getElementById('loading')!.style.display = 'none';
		document.getElementById('unity')!.style.visibility = 'visible';
	}, []);

	useEffect(() => {
		addEventListener('MintAsk', handleMint);
		addEventListener('UnityLoaded', handleLoad);

		return () => {
			removeEventListener('MintAsk', handleMint);
			removeEventListener('UnityLoaded', handleLoad);
		};
	}, [addEventListener, removeEventListener, handleMint, handleLoad]);

	// We'll use a state to store the device pixel ratio.
	const [devicePixelRatio, setDevicePixelRatio] = useState(window.devicePixelRatio);

	const handleChangePixelRatio = useCallback(
		function () {
			// A function which will update the device pixel ratio of the Unity
			// Application to match the device pixel ratio of the browser.
			const updateDevicePixelRatio = function () {
				setDevicePixelRatio(window.devicePixelRatio);
			};
			// A media matcher which watches for changes in the device pixel ratio.
			const mediaMatcher = window.matchMedia(`screen and (resolution: ${devicePixelRatio}dppx)`);
			// Adding an event listener to the media matcher which will update the
			// device pixel ratio of the Unity Application when the device pixel
			// ratio changes.
			mediaMatcher.addEventListener('change', updateDevicePixelRatio);
			return function () {
				// Removing the event listener when the component unmounts.
				mediaMatcher.removeEventListener('change', updateDevicePixelRatio);
			};
		},
		[devicePixelRatio],
	);

	function handleClickConnectWallet() {
		console.log(wallet?.account.address);
		sendMessage('BlockchainAuth', 'AuthThroughBrowser', wallet?.account.address);
		document.getElementById('button_start')!.style.display = 'none';
	}

	async function approoveMint() {
		document.getElementById('approve_warning')!.style.display = 'inline';

		const tx = {
			validUntil: Date.now() + 1000000,
			messages: [
				{
					address: 'EQDpHi-RxVOwEnwlTwR9FRuHRmiI6oop4mExd7CndECkHRNR',
					amount: '50000000',
				},
			],
		};

		await sendTransaction(tx, walletsList.contents.walletsList[0]);

		const requestOptions = {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ owner_addrs: wallet?.account.address }),
		};
		fetch(
			'http://35.246.152.136:8000/api/v1/generate/shield/',
			requestOptions,
		)
			.then((response) => response.json())
			.then((data) => {
				console.log(data['public_url']);
				sendMessage('Main Camera', 'onDeepLinkActivated', 'genlock?' + data['public_url']);
				document.getElementById('button_approve')!.style.display = 'none';
				document.getElementById('approve_warning')!.style.display = 'none';
			});
	}

	return (
		<div className="send-tx-form">
			<Title level={3}>Welcome to Genlock!</Title>

			<div id="loading" style={{ display: 'none', textAlign: 'center' }} >
				<p style={{ color: 'yellow', fontSize: 'large', textAlign: 'center' }} >Loading...</p>
				<table><tbody>
					<tr>
						<th><b>Key</b></th>
						<th><b>Action</b></th>
					</tr>
				
					<tr>
						<td><img src="https://storage.googleapis.com/nft-game-assets/front/wasd.png" width="200"/></td>
						<td>Move</td>
					</tr>

					<tr>
						<td><img src="https://storage.googleapis.com/nft-game-assets/front/left-click.png" width="100"/></td>
						<td>Attack</td>
					</tr>

					<tr>
						<td><img src="https://storage.googleapis.com/nft-game-assets/front/space.png" width="200" height="auto"/></td>
						<td>Jump</td>
					</tr>

					<tr>
						<td><img src="https://storage.googleapis.com/nft-game-assets/front/shift.png" width="100" height="auto"/></td>
						<td>Block</td>
					</tr>
				</tbody></table>
			</div>


			<div id="button_approve" style={{ display: 'none', textAlign: 'center' }}>
				<div id="approve_warning" style={{ textAlign: 'center', display: 'none', color: 'yellow', fontSize: 'large' }}> 
					<p>Open Tonkeeper and approve the transaction</p>
				</div>
				<Button shape="round" onClick={approoveMint}>
					Approve transaction
				</Button>
				<Modal
					title="Connect to Tonkeeper"
					open={!!modalUniversalLink}
					onOk={() => setModalUniversalLink('')}
					onCancel={() => setModalUniversalLink('')}
				>
					<QRCode
						size={256}
						style={{ height: '260px', maxWidth: '100%', width: '100%' }}
						value={modalUniversalLink}
						viewBox={`0 0 256 256`}
					/>
				</Modal>
			</div>

			{wallet ? (
				<>
					
				</>
			) : (
				<>
					<div className="send-tx-form__error">Connect your wallet to start the game</div>
					<Button shape="round" type="primary" onClick={handleButtonClick}>
						Connect Wallet
					</Button>
					<Modal
						title="Connect to Tonkeeper"
						open={!!modalUniversalLink}
						onOk={() => setModalUniversalLink('')}
						onCancel={() => setModalUniversalLink('')}
					>
						<QRCode
							size={256}
							style={{ height: '260px', maxWidth: '100%', width: '100%' }}
							value={modalUniversalLink}
							viewBox={`0 0 256 256`}
						/>
					</Modal>
				</>
			)}
			<div id="unity" style={{ width: '100%', textAlign: 'center', display: 'none', visibility: 'hidden' }}>
				<div  id="button_start" style={{ display: 'inline', textAlign: 'center' }}>
					<Title level={4}>The wallet was successfully connected</Title>
					<br></br>
					<Button shape="round" onClick={handleClickConnectWallet}>
						Start game
					</Button>
					<br></br>
					<br></br>
				</div>
				<Unity
					unityProvider={unityProvider}
					style={{ width: 800, height: 600, margin: 'auto' }}
					devicePixelRatio={devicePixelRatio}
				/>
			</div>
		</div>
	);
}
