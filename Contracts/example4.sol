// SPDX-License-Identifier: MIT
pragma solidity 0.8.0;

/**
 * @title SecureCrowdfund
 * @dev A crowdfunding contract safe from reentrancy.
 */
contract SecureCrowdfund {

    address public immutable owner;
    uint public immutable goal;
    uint public immutable deadline;
    bool public goalMet;
    uint public totalRaised;
    mapping(address => uint) public contributions;

    event Funded(address indexed contributor, uint amount);
    event Refunded(address indexed contributor, uint amount);

    constructor(uint _goalInEther, uint _durationInSeconds) {
        owner = msg.sender;
        goal = _goalInEther * 1 ether;
        deadline = block.timestamp + _durationInSeconds;
    }

    function contribute() public payable {
        require(block.timestamp < deadline, "Campaign has ended");
        require(msg.value > 0, "Contribution must be positive");
        contributions[msg.sender] += msg.value;
        totalRaised += msg.value;
        
        if (totalRaised >= goal) {
            goalMet = true;
        }
        emit Funded(msg.sender, msg.value);
    }

    /**
     * @dev Secure refund function if the goal was not met.
     */
    function claimRefund() public {
        
        // 1. CHECKS
        require(block.timestamp > deadline, "Campaign is still active");
        require(!goalMet, "Campaign goal was met");
        
        uint refundAmount = contributions[msg.sender];
        require(refundAmount > 0, "No contribution to refund");

        // 2. EFFECTS (THE FIX)
        // The contribution is zeroed out *before* the refund is sent.
        contributions[msg.sender] = 0;

        emit Refunded(msg.sender, refundAmount);

        // 3. INTERACTION
        (bool sent, ) = msg.sender.call{value: refundAmount}("");
        require(sent, "Refund transfer failed");
    }
}