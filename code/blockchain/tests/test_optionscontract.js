const OptionsContract = artifacts.require("OptionsContract");

contract("OptionsContract", (accounts) => {
  let instance;
  const ETH_USD_FEED = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419";

  before(async () => {
    instance = await OptionsContract.new(ETH_USD_FEED);
  });

  it("should create valid call options", async () => {
    await instance.createOption(3000, 30, 0, {
      value: ethers.utils.parseEther("1"),
      from: accounts[0],
    });
    const option = await instance.options(0);
    assert.equal(option.holder, accounts[0], "Option creation failed");
  });
});
