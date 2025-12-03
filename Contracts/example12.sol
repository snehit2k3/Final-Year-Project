// SPDX-License-Identifier: MIT
pragma solidity 0.8.0;

/**
 * @title SecureEscrow
 * @dev A 3-party escrow contract that is reentrancy-safe.
 */
contract SecureEscrow {

    address public immutable depositor;
    address public immutable beneficiary;
    address public immutable arbiter;
    
    uint public amount;
    bool public isFunded;
    bool public isReleased;

    event Funded(address indexed depositor, uint amount);
    event Released(address indexed by, address indexed to);

    constructor(address _beneficiary, address _arbiter) {
        depositor = msg.sender;
        beneficiary = _beneficiary;
        arbiter = _arbiter;
    }

    /**
     * @dev The depositor funds the escrow.
     */
    function deposit() public payable {
        require(msg.sender == depositor, "Only depositor can fund");
        require(!isFunded, "Escrow already funded");
        
        amount = msg.value;
        isFunded = true;
        
        emit Funded(depositor, amount);
    }

    /**
     * @dev Arbiter or Depositor can release funds to the beneficiary.
     */
    function releaseFunds() public {
        
        // 1. CHECKS
        require(msg.sender == arbiter || msg.sender == depositor, "Not authorized");
        require(isFunded, "Escrow not funded");
        require(!isReleased, "Funds already released");

        // 2. EFFECTS (THE FIX)
        // The state is updated to "released" *before* the call.
        isReleased = true;

        emit Released(msg.sender, beneficiary);

        // 3. INTERACTION
        // Note: The call is to the 'beneficiary', not 'msg.sender'.
        // The CEI pattern still protects the contract's state.
        (bool sent, ) = beneficiary.call{value: amount}("");
        require(sent, "Failed to release funds");
    }
}