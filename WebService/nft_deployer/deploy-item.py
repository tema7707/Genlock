import sys

from nft_deployer.pyfift.base.app import App
from nft_deployer.pyfift.wallet.wallet_v3_r2 import WalletV3R2
from nft_deployer.pyfift.nft.nft_deploy import DeployNFTMessage


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
    print("insufficient balance for sending nft item message, min: 0.05 TON")
    sys.exit()


nft_collection_addr = "kQCwb6d8h1e6al73SbHwfPM7rmidYUwkEvVfgnykXG5YfAb4"
msg_body = DeployNFTMessage(
    index=0,
    content_url="/home/dkrivenkov/program/genlock/my_nft.json",
    amount=50000000,
    owner="kQCg9BqmY4SqNw3GLNZ0QAA4e74QKS33Tm9v_6WFROaKiYEP",
).to_boc()
wallet.send_to_contract(msg_body, 50000000, nft_collection_addr)