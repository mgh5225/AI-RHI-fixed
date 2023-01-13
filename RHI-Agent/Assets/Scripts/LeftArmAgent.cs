using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.SceneManagement;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class representing the Agent.
// ===============================
public class LeftArmAgent : Agent
{
    /// <summary>
    /// Parameters that allow the specification of the joint angles
    /// </summary>
    [Tooltip("Left shoulder joint")]
    public GameObject leftShoulderJoint;

    [Tooltip("Left upper arm joint")]
    public GameObject leftUpperArmJoint;

    [Tooltip("Left elbow joint")]
    public GameObject leftElbowJoint;

    [Tooltip("Head joint")]
    public GameObject headJoint;

    /// <summary>
    /// Parameters that allow for distance measurement
    /// </summary>
    [Tooltip("Middle hand")]
    public GameObject middleHand;

    [Tooltip("Rubber Arm")]
    public GameObject rubberArm;

    /// <summary>
    /// Parameters that allow for the perception of visuo-tactile stimulation event times
    /// </summary>
    [Tooltip("Ball")]
    public GameObject ballHandler;

    [Tooltip("Vib")]
    public GameObject vibHandler;


    /// <summary>
    /// Joint controller objects for the joints.
    /// </summary>
    private JointController leftShoulder;
    private JointController leftUpperArm;
    private JointController leftElbow;
    private JointController head;

    private RubberArmController rubberArmController;
    private BallHandler ballScript;
    private VibHandler vibScript;

    /// <summary>
    /// Angular velocity multiplier
    /// Velocity in degrees/second of the joint angles when input action == 1.
    /// </summary>
    private float turnSpeed = 1f;

    /// <summary>
    /// Initial setup, called at startup.
    /// </summary>
    /// <remarks>
    /// In future versions of ML-agents (> 0.14), InitializeAgent should be replaced by Initialize.
    /// </remarks>
    public override void Initialize()
    {
        base.Initialize();

        // Initialize joint controllers
        leftShoulder = new JointController("Left shoulder", leftShoulderJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999)); //new Vector3(-999, -999, -25), new Vector3(999, 999, 25));
        leftUpperArm = new JointController("Left upper arm", leftUpperArmJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999));
        leftElbow = new JointController("Left elbow", leftElbowJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999)); //new Vector3(-25, -999, -999), new Vector3(25, 999, 999));
        head = new JointController("Head", headJoint, new Vector3(-999, -999, -999), new Vector3(999, 999, 999));

        // Retrieve the rubberArm and visuo-tactile stimulation objects from the environment
        rubberArmController = rubberArm.GetComponent<RubberArmController>();
        ballScript = ballHandler.GetComponent<BallHandler>();
        vibScript = vibHandler.GetComponent<VibHandler>();

        // Set the head rotation (and subsequently the camera perspective)
        head.SetRelativeJointAngles(new Vector3(15f, -20f));
    }

    /// <summary>
    /// Perform actions based on a vector of numbers
    /// </summary>
    /// <remarks>
    /// In future versions of ML-agents (> 0.14), AgentAction should be replaced by OnActionReceived.
    /// </remarks>
    /// <param name="vectorAction">The list of actions to take</param>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var vectorAction = actionBuffers.ContinuousActions;
        // The first action vector value specifies the type of control from the Python environment
        switch (vectorAction[0])
        {
            // Normal action (joint velocity control)
            default:
                leftShoulder.MoveJoint(0f, 0f, vectorAction[1] * turnSpeed * Time.fixedDeltaTime);
                leftElbow.MoveJoint(vectorAction[2] * turnSpeed * Time.fixedDeltaTime, 0f, 0f);
                break;

            // Joint angle rotation (set joint angle directly)
            case 1f:
                leftShoulder.SetRelativeJointAngles(new Vector3(0f, 0f, vectorAction[1]));
                leftElbow.SetRelativeJointAngles(new Vector3(vectorAction[2], 0f, 0f));
                break;

            // Rubber arm joint (set rubber arm joint angle)
            case 2f:
                rubberArmController.setRelativeLeftShoulderZ(vectorAction[1]);
                rubberArmController.setRelativeLeftElbowX(vectorAction[2]);
                break;

            case 3f:
                ballScript.changeActiveBallY(vectorAction[1]);
                break;
        }
    }

    /// <summary>
    /// Read inputs from the keyboard and convert them to a list of actions.
    /// Called when the user wants to control the agent directly by setting
    /// Behavior Type to "Heuristic Only" in the Behavior Parameters inspector (of the Real body object).
    /// </summary>
    /// <remarks>
    /// Control:
    ///     W -> Extend elbow
    ///     S -> Flex elbow (bring lower arm to upper arm)
    ///     A -> Abduct shoulder (move arm away from body)
    ///     D -> Adduct shoulder (move arm towards body)
    /// </remarks>
    /// <returns>A vector of action that will be passed to AgentAction(float[])</returns>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        float shoulderRotate = 0f;
        if (Input.GetKey(KeyCode.A))
        {
            shoulderRotate = -0.5f;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            shoulderRotate = 0.5f;
        }

        float elbowRotate = 0f;
        if (Input.GetKey(KeyCode.W))
        {
            elbowRotate = -0.5f;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            elbowRotate = 0.5f;
        }

        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = 0f;
        continuousActionsOut[1] = shoulderRotate;
        continuousActionsOut[2] = elbowRotate;
    }

    /// <summary>
    /// Called upon reset of the environment. Resets the joint positions.
    /// </summary>
    /// <remarks>
    /// In future versions of ML-agents (> 0.14), AgentReset should be replaced by OnEpisodeBegin.
    /// </remarks>
    public override void OnEpisodeBegin()
    {
        leftShoulder.ResetJoint();
        leftUpperArm.ResetJoint();
        leftElbow.ResetJoint();
    }


    /// <summary>
    /// Collect observations. Number of calls to AddVectorObs should match with the
    /// observation vector size set in the Behavior Parameters inspector (of the Real body object).
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Shoulder extension/flexion
        sensor.AddObservation(leftShoulder.GetRelativeZ());

        // Elbow extension/flexion
        sensor.AddObservation(leftElbow.GetRelativeX());

        // Hand touch state (visual)
        sensor.AddObservation(ballScript.GetLastTouch());

        // Hand touch state (tactile)
        sensor.AddObservation(vibScript.GetLastVib());

        // Current Time
        sensor.AddObservation(Time.time);

        // Absolute hand distance error for logging
        sensor.AddObservation(Vector3.Distance(middleHand.transform.position, rubberArmController.getMiddleHand().transform.position));

        // Horizontal distance
        sensor.AddObservation(middleHand.transform.position.z - rubberArmController.getMiddleHand().transform.position.z);

        // Shoulder extension/flexion
        sensor.AddObservation(rubberArmController.getRelativeLeftShoulderZ());

        // Elbow extension/flexion
        sensor.AddObservation(rubberArmController.getRelativeLeftElbowX());

        var active_ball = ballScript.getActiveBall();
        float distance = 0;
        if (active_ball != null)
            distance = Vector3.Distance(active_ball.transform.position, leftElbowJoint.transform.position);

        sensor.AddObservation(distance);

    }

}