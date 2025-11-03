// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title VulnerableSavingsPot
 * @dev THIS CONTRACT IS INTENTIONALLY VULNERABLE TO REENTRANCY. DO NOT USE IN PRODUCTION.
 * This contract simulates a savings pot where users can deposit ETH and withdraw it later.
 * It has a flaw in the withdraw function where the external call to send Ether is made
 * before the user's balance is set to zero. An attacker can exploit this by having their
 * fallback function call withdraw() again, draining the contract's funds.
 */
contract VulnerableSavingsPot {
    // Mapping of user addresses to their balances
    mapping(address => uint256) public userBalances;

    // Mapping to track deposit timestamps for a pseudo-reward system
    mapping(address => uint256) public depositTimestamps;

    // A small reward rate for holding funds
    uint256 public constant REWARD_RATE_PER_SECOND = 0.00001 ether;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    /**
     * @dev Allows a user to deposit Ether into the contract.
     */
    function deposit() external payable {
        require(msg.value > 0, "Deposit amount must be greater than zero.");
        userBalances[msg.sender] += msg.value;
        if (depositTimestamps[msg.sender] == 0) {
            depositTimestamps[msg.sender] = block.timestamp;
        }
        emit Deposited(msg.sender, msg.value);
    }

    /**
     * @dev Calculates a simple time-based reward. This is just for demonstration.
     */
    function calculateReward(address user) public view returns (uint256) {
        if (depositTimestamps[user] == 0) {
            return 0;
        }
        uint256 timeHeld = block.timestamp - depositTimestamps[user];
        return timeHeld * REWARD_RATE_PER_SECOND;
    }

    /**
     * @dev VULNERABLE WITHDRAW FUNCTION
     * Allows a user to withdraw their entire balance plus any calculated rewards.
     * The vulnerability is that it sends ETH (interaction) before updating the state (effect).
     */
    function withdraw() external {
        uint256 balance = userBalances[msg.sender];
        require(balance > 0, "You have no funds to withdraw.");

        uint256 reward = calculateReward(msg.sender);
        uint256 totalWithdrawAmount = balance + reward;

        // --- THE VULNERABILITY IS HERE ---
        // 1. (Interaction) The contract sends Ether to the msg.sender.
        //    If msg.sender is a malicious contract, its receive() or fallback()
        //    function is triggered.
        (bool success, ) = msg.sender.call{value: totalWithdrawAmount}("");
        require(success, "Failed to send Ether.");

        // 2. (Effect) The user's balance is updated *after* the external call.
        //    The attacker's contract can call withdraw() again before this line is reached,
        //    and since the balance is not yet zero, the check passes again.
        userBalances[msg.sender] = 0;
        depositTimestamps[msg.sender] = 0; // Reset timestamp

        emit Withdrawn(msg.sender, totalWithdrawAmount);
    }

    /**
     * @dev A helper function to check the balance of this contract.
     */
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}