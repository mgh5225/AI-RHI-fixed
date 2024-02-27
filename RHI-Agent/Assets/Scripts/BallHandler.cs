using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

// ===============================
// AUTHOR: Thomas Rood
// PURPOSE: Class that handles the activation of the correct Ball object and keeps track of the last (visual) Ball touch event time.
// SPECIAL NOTES: Note that the environment contains a separate Ball object for each condition, to visually place it in the right position.
// ===============================
public class BallHandler : MonoBehaviour
{
    [Tooltip("Parameters object")]
    public GameObject parameterObject;

    public Transform ballPosition;

    private Parameters parameterScript;
    private GameObject ball_l;
    private GameObject ball_c;
    private GameObject ball_r;
    private GameObject ball_d;
    private float lastTouch;
    private object lastTouchLock = new object();

    /// <summary>
    /// Start is called before the first frame.
    /// The function calls SetActiveBall to activate the correct ball and 
    /// makes sure this function is called upon every reset of the environment in case the parameters changed.
    /// </summary>
    void Start()
    {
        Academy.Instance.OnEnvironmentReset += () =>
        {
            SetActiveBall();
        };

        SetActiveBall();

        lastTouch = -60f;
    }

    /// <summary>
    /// Activates the correct ball object according to the parameter setting.
    /// </summary>
    private void SetActiveBall()
    {
        parameterScript = parameterObject.GetComponent<Parameters>();

        ball_l = this.transform.Find("ball_l").gameObject;
        ball_c = this.transform.Find("ball_c").gameObject;
        ball_r = this.transform.Find("ball_r").gameObject;
        ball_d = this.transform.Find("ball_d").gameObject;

        if (parameterScript.mode == Parameters.Mode.dataGenerationWithBall)
        {
            ball_l.SetActive(parameterScript.condition == Parameters.Condition.Left);
            ball_c.SetActive(parameterScript.condition == Parameters.Condition.Center);
            ball_r.SetActive(parameterScript.condition == Parameters.Condition.Right);

            ball_l.GetComponent<Animator>().enabled = false;
            ball_c.GetComponent<Animator>().enabled = false;
            ball_r.GetComponent<Animator>().enabled = false;
        }
        else
        {
            ball_d.SetActive(true);
            SetRange(
                ball_d,
                ballPosition.position,
                parameterScript.ballRange.b_min,
                parameterScript.ballRange.b_max
            );
        }
    }

    public void SetRange(GameObject obj, Vector3 pos, float min, float max)
    {
        var _animation = obj.GetComponent<Animation>();

        if (!_animation) _animation = obj.AddComponent<Animation>();

        var clip = new AnimationClip();

        clip.name = "Dynamic Ball";
        clip.legacy = true;

        var k_x = new Keyframe[1];
        var k_y = new Keyframe[3];
        var k_z = new Keyframe[1];

        k_x[0] = new Keyframe(0f, pos.x - transform.position.x);

        k_y[0] = new Keyframe(0f, max + pos.y - transform.position.y + 0.025f);
        k_y[1] = new Keyframe(1f, min + pos.y - transform.position.y + 0.025f);
        k_y[1].weightedMode = WeightedMode.Both;
        k_y[1].inTangent = -1f;
        k_y[1].outTangent = 1f;
        k_y[1].inWeight = max - min;
        k_y[1].outWeight = max - min;
        k_y[2] = new Keyframe(2f, max + pos.y - transform.position.y + 0.025f);

        k_z[0] = new Keyframe(0f, pos.z - transform.position.z);

        var x_curve = new AnimationCurve(k_x);
        var y_curve = new AnimationCurve(k_y);
        var z_curve = new AnimationCurve(k_z);

        clip.SetCurve("", typeof(Transform), "localPosition.x", x_curve);
        clip.SetCurve("", typeof(Transform), "localPosition.y", y_curve);
        clip.SetCurve("", typeof(Transform), "localPosition.z", z_curve);

        var evt = new AnimationEvent();

        evt.functionName = "TouchCallBack";
        evt.time = 1f;

        clip.AddEvent(evt);

        _animation.clip = clip;
        _animation.AddClip(clip, clip.name);
        _animation.playAutomatically = true;
        _animation.wrapMode = WrapMode.Loop;
        _animation.Play();
    }

    /// <summary>
    /// Called when the Ball reaches the lowest point of the animation.
    /// Sets the last (visual) touch time.
    /// </summary>
    /// <param name="touchTime"></param>
    public void TouchCallBack(float touchTime)
    {
        lock (lastTouchLock)
        {
            lastTouch = touchTime;
        }
    }

    /// <summary>
    /// Get the last (visual) touch event time.
    /// </summary>
    /// <returns>float lastTouch</returns>
    public float GetLastTouch()
    {
        lock (lastTouchLock)
        {
            return lastTouch;
        }
    }

    public GameObject getActiveBall()
    {
        GameObject active_ball = null;
        if (ball_l.activeSelf == true) active_ball = ball_l;
        if (ball_c.activeSelf == true) active_ball = ball_c;
        if (ball_r.activeSelf == true) active_ball = ball_r;
        if (ball_d.activeSelf == true) active_ball = ball_d;

        return active_ball;
    }

    public void changeActiveBallY(float yAxis)
    {
        var active_ball = getActiveBall();
        if (active_ball == null) return;

        active_ball.transform.position = Vector3.Scale(active_ball.transform.position, Vector3.right + Vector3.forward) + new Vector3(0, yAxis, 0);

    }

}
