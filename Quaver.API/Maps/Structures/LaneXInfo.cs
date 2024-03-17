/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * Copyright (c) 2017-2019 Swan & The Quaver Team <support@quavergame.com>.
*/

using System;
using System.Collections.Generic;
using MoonSharp.Interpreter;
using MoonSharp.Interpreter.Interop;
using YamlDotNet.Serialization;

namespace Quaver.API.Maps.Structures
{
    /// <summary>
    ///     SliderVelocities section of the .qua
    /// </summary>
    [Serializable]
    [MoonSharpUserData]
    public class LaneXInfo
    {
        /// <summary>
        ///     The time in milliseconds when the new SliderVelocity section begins
        /// </summary>
        public float Time
        {
            get;
            [MoonSharpVisible(false)] set;
        }

        /// <summary>
        ///     The velocity multiplier relative to the current timing section's BPM
        /// </summary>
        public float PositionX
        {
            get;
            [MoonSharpVisible(false)] set;
        }
        
        public int Lane { get; [MoonSharpVisible(false)] set; }

        /// <summary>
        ///     Returns if the SV is allowed to be edited in lua scripts
        /// </summary>
        [YamlIgnore]
        public bool IsEditableInLuaScript
        {
            get;
            [MoonSharpVisible(false)] set;
        }

        /// <summary>
        ///     Sets the start time of the SV.
        ///     FOR USE IN LUA SCRIPTS ONLY.
        /// </summary>
        /// <param name="time"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public void SetTime(float time)
        {
            ThrowUneditableException();
            Time = time;
        }

        /// <summary>
        ///     Sets the multiplier of the SV.
        ///     FOR USE IN LUA SCRIPTS ONLY.
        /// </summary>
        /// <param name="x"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public void SetPositionX(float x)
        {
            ThrowUneditableException();
            PositionX = x;
        }
        
        public void SetLane(int lane)
        {
            ThrowUneditableException();
            Lane = lane;
        }

        /// <summary>
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
        private void ThrowUneditableException()
        {
            if (!IsEditableInLuaScript)
                throw new InvalidOperationException("Value is not allowed to be edited in lua scripts.");
        }

        private sealed class ByValueEqualityComparer : IEqualityComparer<LaneXInfo>
        {
            public bool Equals(LaneXInfo x, LaneXInfo y)
            {
                if (ReferenceEquals(x, y)) return true;
                if (ReferenceEquals(x, null)) return false;
                if (ReferenceEquals(y, null)) return false;
                if (x.GetType() != y.GetType()) return false;
                return x.Time.Equals(y.Time) && x.PositionX.Equals(y.PositionX) && x.Lane == y.Lane;
            }

            public int GetHashCode(LaneXInfo obj)
            {
                return HashCode.Combine(obj.Time, obj.PositionX, obj.Lane);
            }
        }

        public static IEqualityComparer<LaneXInfo> ByValueComparer { get; } = new ByValueEqualityComparer();
    }
}
