// SPDX-License-Identifier: MIT
// WARNING: VULNERABLE CODE!
pragma solidity ^0.8.0;

/**
 * @title VulnerableRewards
 * @dev This contract manages user stakes and rewards.
 *
 * THE FLAW: A Cross-Function Reentrancy.
 * `claimRewards` has a classic reentrancy flaw.
 * An attacker's fallback function can call `compoundRewards`.
 * Since both functions read/write the 'rewards' mapping,
 * the attacker can claim their ETH *and* compound the same
 * rewards in a single transaction.
 */
contract VulnerableRewards {

    mapping(address => uint) public stakes;
    mapping(address => uint) public rewards;
    address public owner;
    
    event Claimed(address indexed user, uint amount);
    event Compounded(address indexed user, uint amount);

    constructor() {
        owner = msg.sender;
    }

    // --- Admin Functions ---
    /**
     * @dev Owner can "airdrop" rewards to a user (for demo purposes).
     */
    function addRewards(address _user, uint _amount) public {
        require(msg.sender == owner, "Not owner");
        rewards[_user] += _amount;
    }

    // --- User Functions ---
    /**
     * @dev VULNERABLE function to withdraw pending rewards as Ether.
     */
    function claimRewards() public {
        // 1. CHECK
        uint rewardAmount = rewards[msg.sender];
        require(rewardAmount > 0, "No rewards to claim");

        // 2. INTERACTION (THE FLAW)
        // Ether is sent *before* the reward balance is zeroed.
        // Attacker's fallback is triggered here.
        (bool sent, ) = msg.sender.call{value: rewardAmount}("");
        require(sent, "Failed to send reward");

        // 3. EFFECT (TOO LATE)
        // 'rewards' mapping is only updated after the external call.
        rewards[msg.sender] = 0;
        
        emit Claimed(msg.sender, rewardAmount);
    }

    /**
     * @dev Function to re-invest rewards into the user's stake.
     * This is the "cross-function" that will be called by the attacker.
     */
    function compoundRewards() public {
        // 1. CHECK
        uint rewardAmount = rewards[msg.sender];
        require(rewardAmount > 0, "No rewards to compound");

        // 2. EFFECTS
        // This function is internally "safe", but it can be
        // exploited by the vulnerability in `claimRewards`.
        rewards[msg.sender] = 0;
        stakes[msg.sender] += rewardAmount;
        
        emit Compounded(msg.sender, rewardAmount);
    }
}