// SPDX-License-Identifier: MIT
// WARNING: VULNERABLE CODE!
pragma solidity ^0.8.0;

/**
 * @title VulnerableCrowdfund
 * @dev A contract to raise a target amount of Ether by a deadline.
 * If the goal is not met, contributors can withdraw their funds.
 *
 * THE FLAW: `claimRefund` sends Ether (Interaction) before
 * updating the user's contribution balance (Effect).
 */
contract VulnerableCrowdfund {

    address public owner;
    uint public immutable goal;
    uint public immutable deadline;
    bool public goalMet;
    
    mapping(address => uint) public contributions;
    uint public totalRaised;

    event Contribution(address indexed contributor, uint amount);
    event Refunded(address indexed contributor, uint amount);
    event GoalMet(uint total);

    constructor(uint _goalInEther, uint _durationInSeconds) {
        owner = msg.sender;
        goal = _goalInEther * 1 ether;
        deadline = block.timestamp + _durationInSeconds;
    }

    /**
     * @dev Allows users to contribute to the fund.
     */
    function contribute() public payable {
        require(block.timestamp < deadline, "Campaign has ended");
        require(!goalMet, "Campaign goal has already been met");
        
        contributions[msg.sender] += msg.value;
        totalRaised += msg.value;
        
        emit Contribution(msg.sender, msg.value);

        if (totalRaised >= goal) {
            goalMet = true;
            emit GoalMet(totalRaised);
        }
    }

    /**
     * @dev The VULNERABLE refund function.
     * Allows users to get their money back if the goal was not met.
     */
    function claimRefund() public {
        // 1. CHECKS
        require(block.timestamp > deadline, "Campaign is still active");
        require(!goalMet, "Campaign goal was met");
        
        uint refundAmount = contributions[msg.sender];
        require(refundAmount > 0, "No contribution to refund");

        // 2. INTERACTION (THE FLAW)
        // Ether is sent *before* the contribution amount is set to 0.
        // An attacker's 'receive()' function can call 'claimRefund()'
        // again, passing the checks on lines 60-61 repeatedly.
        (bool sent, ) = msg.sender.call{value: refundAmount}("");
        require(sent, "Refund transfer failed");

        // 3. EFFECT (TOO LATE)
        // The attacker has already drained their 'refundAmount'
        // multiple times before this line is ever reached.
        contributions[msg.sender] = 0;

        emit Refunded(msg.sender, refundAmount);
    }

    /**
     * @dev Allows the owner to withdraw funds if the goal was met.
     */
    function ownerWithdraw() public {
        require(msg.sender == owner, "Not owner");
        require(goalMet, "Goal not met");
        
        (bool sent, ) = owner.call{value: address(this).balance}("");
        require(sent, "Withdrawal failed");
    }
}