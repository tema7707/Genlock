import sys

from nft_deployer.pyfift.base.app import App
from nft_deployer.pyfift.wallet.wallet_v3_r2 import WalletV3R2
from nft_deployer.pyfift.nft.nft_collection import NftCollection


nft_config = "/home/dkrivenkov/program/genlock/configs/nft/config.json"
App.init(config=nft_config)

wallet = WalletV3R2()
wallet.init_data()
addr = wallet.address(binary=False)
state = App.lite_client.state(addr)

if state["state"] != "active":
    if state["state"] == "empty": print("Empty account, send some TONs and deploy it ...")
    elif state["state"] == "inactive": print("Deploy the wallet contract before proceeding ...")
    sys.exit()

if state["balance"] < 50000000:
    print("insufficient balance for deploying nft contract, min: 0.05 TON")
    sys.exit()

collection = NftCollection()
collection.init_data(
    owner=addr,
    royalty_factor=1,
    royalty_base=100,
    next_item_index=0,
    collection_content_url='https://storage.googleapis.com/nft-game-assets/collection.json',
    common_content_url='https://storage.googleapis.com/nft-game-assets/metadata/',
)
collection.prepare_deploy(value=0.05, external=False)
print("preparing to deploy nft collection contract ...")
print("NFT Collection address:", collection.h_addr)
collection.deploy(wallet, mode=64 + 3)
