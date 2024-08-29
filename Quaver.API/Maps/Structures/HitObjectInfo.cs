/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * Copyright (c) 2017-2019 Swan & The Quaver Team <support@quavergame.com>.
*/

using System;
using System.Collections.Generic;
using System.Linq;
using MoonSharp.Interpreter;
using MoonSharp.Interpreter.Interop;
using Quaver.API.Enums;
using Quaver.API.Helpers;
using YamlDotNet.Serialization;

namespace Quaver.API.Maps.Structures
{
    /// <summary>
    ///     HitObjects section of the .qua
    /// </summary>
    [MoonSharpUserData]
    [Serializable]
    public class HitObjectInfo : IStartTime
    {
        /// <summary>
        ///     The time in milliseconds when the HitObject is supposed to be hit.
        /// </summary>
        public int StartTime
        {
            get;
            [MoonSharpVisible(false)]
            set;
        }

        /// <summary>
        ///     The lane the HitObject falls in
        /// </summary>
        public int Lane
        {
            get;
            [MoonSharpVisible(false)]
            set;
        }

        /// <summary>
        ///     The endtime of the HitObject (if greater than 0, it's considered a hold note.)
        /// </summary>
        public int EndTime { get; [MoonSharpVisible(false)] set; }

        float IStartTime.StartTime
        {
            get => StartTime;
            set => StartTime = (int)value;
        }

        /// <summary>
        ///     Bitwise combination of hit sounds for this object
        /// </summary>
        public HitSounds HitSound
        {
            get;
            [MoonSharpVisible(false)]
            set;
        }

        /// <summary>
        ///     Key sounds to play when this object is hit.
        /// </summary>
        [MoonSharpVisible(false)]
        public List<KeySoundInfo> KeySounds { get; set; } = new List<KeySoundInfo>();

        /// <summary>
        ///     The layer in the editor that the object belongs to.
        /// </summary>
        public int EditorLayer
        {
            get;
            [MoonSharpVisible(false)]
            set;
        }

        /// <summary>
        ///     If the object is a long note. (EndTime > 0)
        /// </summary>
        [YamlIgnore]
        public bool IsLongNote => EndTime > 0;

        /// <summary>
        ///     Returns if the object is allowed to be edited in lua scripts
        /// </summary>
        [YamlIgnore]
        public bool IsEditableInLuaScript
        {
            get;
            [MoonSharpVisible(false)]
            set;
        }

        /// <summary>
        ///     Gets the timing point this object is in range of.
        /// </summary>
        /// <returns></returns>
        public TimingPointInfo GetTimingPoint(List<TimingPointInfo> timingPoints) =>
            timingPoints.AtTime(StartTime) ?? timingPoints[0];

        /// <summary>
        /// </summary>
        /// <param name="time"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public void SetStartTime(int time)
        {
            ThrowUneditableException();
            StartTime = time;
        }

        /// <summary>
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
        public void SetEndTime(int time)
        {
            ThrowUneditableException();
            EndTime = time;
        }

        /// <summary>
        /// </summary>
        /// <param name="lane"></param>
        public void SetLane(int lane)
        {
            ThrowUneditableException();
            Lane = lane;
        }

        /// <summary>
        /// </summary>
        /// <param name="hitsounds"></param>
        public void SetHitSounds(HitSounds hitsounds)
        {
            ThrowUneditableException();
            HitSound = hitsounds;
        }

        /// <summary>
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
        private void ThrowUneditableException()
        {
            if (!IsEditableInLuaScript)
                throw new InvalidOperationException("Value is not allowed to be edited in lua scripts.");
        }

        /// <summary>
        ///     By-value comparer, mostly auto-generated by Rider: KeySounds-related code is changed to by-value.
        /// </summary>
        private sealed class ByValueEqualityComparer : IEqualityComparer<HitObjectInfo>
        {
            public bool Equals(HitObjectInfo x, HitObjectInfo y)
            {
                if (ReferenceEquals(x, y)) return true;
                if (ReferenceEquals(x, null)) return false;
                if (ReferenceEquals(y, null)) return false;
                if (x.GetType() != y.GetType()) return false;

                return x.StartTime == y.StartTime &&
                    x.Lane == y.Lane &&
                    x.EndTime == y.EndTime &&
                    x.HitSound == y.HitSound &&
                    x.KeySounds.SequenceEqual(y.KeySounds, KeySoundInfo.ByValueComparer) &&
                    x.EditorLayer == y.EditorLayer;
            }

            public int GetHashCode(HitObjectInfo obj)
            {
                unchecked
                {
                    var hashCode = obj.StartTime;
                    hashCode = (hashCode * 397) ^ obj.Lane;
                    hashCode = (hashCode * 397) ^ obj.EndTime;
                    hashCode = (hashCode * 397) ^ (int)obj.HitSound;

                    foreach (var keySound in obj.KeySounds)
                        hashCode = (hashCode * 397) ^ KeySoundInfo.ByValueComparer.GetHashCode(keySound);

                    hashCode = (hashCode * 397) ^ obj.EditorLayer;
                    return hashCode;
                }
            }
        }

        public static IEqualityComparer<HitObjectInfo> ByValueComparer { get; } = new ByValueEqualityComparer();
    }
}
