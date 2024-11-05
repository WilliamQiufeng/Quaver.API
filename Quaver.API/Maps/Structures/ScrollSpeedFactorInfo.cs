using System;
using System.Collections.Generic;
using System.Text;
using MoonSharp.Interpreter;

namespace Quaver.API.Maps.Structures
{
    /// <summary>
    ///     ScrollSpeedFactors section of the .qua
    /// </summary>
    [Serializable]
    [MoonSharpUserData]
    public class ScrollSpeedFactorInfo : IStartTime
    {
        /// <summary>
        ///     The time in milliseconds at which the scroll speed factor will be exactly <see cref="Factor"/>
        /// </summary>
        public float StartTime { get; [MoonSharpHidden] set; }

        /// <summary>
        ///     The multiplier given to the scroll speed.
        ///     It will be lerped to the next scroll speed factor like keyframes, unless this is the last factor change.
        /// </summary>
        public float Factor { get; [MoonSharpHidden] set; }

        /// <summary>
        ///     Bit flag: i-th bit for if this factor applies to the i-th lane. For i >= KeyCount, i-th bit is 1.
        ///     So the value would be -1 to apply to all lanes, even after we change the key count in editor.
        /// </summary>
        public int LaneMask { get; [MoonSharpHidden] set; } = -1;

        public IEnumerable<int> GetLaneMaskLanes(int keyCount) => GetLaneMaskLanes(LaneMask, keyCount);

        public string MaskRepresentation(int keyCount) => MaskRepresentation(LaneMask, keyCount);

        public static IEnumerable<int> GetLaneMaskLanes(int laneMask, int keyCount)
        {
            for (var i = 0; i < keyCount; i++)
            {
                if ((laneMask & (1 << i)) != 0)
                    yield return i;
            }
        }

        /// <summary>
        ///     Mask in human-readable form should be in reverse order and binary, so that the left-most character
        ///     represents the left-most lane (lane 1)
        /// </summary>
        /// <param name="laneMask"></param>
        /// <param name="keyCount"></param>
        /// <returns></returns>
        public static string MaskRepresentation(int laneMask, int keyCount)
        {
            var sb = new StringBuilder();
            for (var i = 0; i < keyCount; i++)
            {
                sb.Append((laneMask & (1 << i)) == 0 ? '0' : '1');
            }

            return sb.ToString();
        }

        /// <summary>
        ///     Turns representation of lane mask back into integer
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static int ParseMask(string input)
        {
            var result = -1;
            for (var i = 0; i < input.Length; i++)
            {
                if (input[i] == '0') result &= ~(1 << i);
            }

            return result;
        }

        private sealed class StartTimeRelationalComparer : IComparer<ScrollSpeedFactorInfo>
        {
            public int Compare(ScrollSpeedFactorInfo x, ScrollSpeedFactorInfo y)
            {
                if (ReferenceEquals(x, y)) return 0;
                if (ReferenceEquals(null, y)) return 1;
                if (ReferenceEquals(null, x)) return -1;
                return x.StartTime.CompareTo(y.StartTime);
            }
        }

        public static IComparer<ScrollSpeedFactorInfo> StartTimeComparer { get; } = new StartTimeRelationalComparer();

        /// <summary>
        ///     By-value comparer, auto-generated by Rider.
        /// </summary>
        private sealed class ByValueEqualityComparer : IEqualityComparer<ScrollSpeedFactorInfo>
        {
            public bool Equals(ScrollSpeedFactorInfo x, ScrollSpeedFactorInfo y)
            {
                if (ReferenceEquals(x, y)) return true;
                if (ReferenceEquals(x, null)) return false;
                if (ReferenceEquals(y, null)) return false;
                if (x.GetType() != y.GetType()) return false;
                return x.StartTime.Equals(y.StartTime) && x.Factor.Equals(y.Factor) && x.LaneMask == y.LaneMask;
            }

            public int GetHashCode(ScrollSpeedFactorInfo obj)
            {
                return HashCode.Combine(obj.StartTime, obj.Factor, obj.LaneMask);
            }
        }

        public static IEqualityComparer<ScrollSpeedFactorInfo> ByValueComparer { get; } = new ByValueEqualityComparer();
    }
}