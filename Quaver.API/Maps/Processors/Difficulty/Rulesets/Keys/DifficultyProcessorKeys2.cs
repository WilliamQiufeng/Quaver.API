/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * Copyright (c) 2017-2018 Swan & The Quaver Team <support@quavergame.com>.
 */

using Quaver.API.Enums;
using Quaver.API.Helpers;
using Quaver.API.Maps;
using Quaver.API.Maps.Processors.Difficulty.Optimization;
using Quaver.API.Maps.Processors.Difficulty.Rulesets.Keys.Structures;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Quaver.API.Maps.Processors.Difficulty.Rulesets.Keys
{
    /// <summary>
    ///     Will be used to solve Strain Rating.
    /// </summary>
    public class DifficultyProcessorKeys : DifficultyProcessor
    {
        /// <summary>
        ///     The version of the processor.
        /// </summary>
        public static string Version { get; } = "0.0.5";

        /// <summary>
        ///     Constants used for solving
        /// </summary>
        public StrainConstantsKeys StrainConstants { get; private set; }

        /// <summary>
        ///     Hit objects in the map used for solving difficulty
        /// </summary>
        public List<StrainSolverData> StrainSolverData { get; private set; } = new List<StrainSolverData>();

        /// <summary>
        ///     Value of confidence that there's vibro manipulation in the calculated map.
        /// </summary>
        private float VibroInaccuracyConfidence { get; set; }

        /// <summary>
        ///     Value of confidence that there's roll manipulation in the calculated map.
        /// </summary>
        private float RollInaccuracyConfidence { get; set; }

        /// <summary>
        ///     Solves the difficulty of a .qua file
        /// </summary>
        /// <param name="map"></param>
        /// <param name="constants"></param>
        /// <param name="mods"></param>
        /// <param name="detailedSolve"></param>
        public DifficultyProcessorKeys(Qua map, StrainConstants constants, ModIdentifier mods = ModIdentifier.None,
            bool detailedSolve = false) : base(map, constants, mods)
        {
            // Cast the current Strain Constants Property to the correct type.
            StrainConstants = (StrainConstantsKeys)constants;

            // Don't bother calculating map difficulty if there's less than 2 hit objects
            if (map.HitObjects.Count < 2)
                return;

            // Solve for difficulty
            CalculateDifficulty(mods);

            // If detailed solving is enabled, expand calculation
            if (detailedSolve)
            {
                // ComputeNoteDensityData();
                ComputeForPatternFlags();
            }
        }

        /// <summary>
        ///     Calculate difficulty of a map with given rate
        /// </summary>
        /// <param name="rate"></param>
        public void CalculateDifficulty(ModIdentifier mods)
        {
            // If map does not exist, ignore calculation.
            if (Map == null) return;

            // Get song rate from selected mods
            var rate = ModHelper.GetRateFromMods(mods);
            var notes = new List<Note>();
            // Add hit objects from qua map to qssData
            for (var i = 0; i < Map.HitObjects.Count; i++)
            {
                if (Map.HasScratchKey && Map.HitObjects[i].Lane == Map.GetKeyCount())
                    continue;

                var curHitOb = Map.HitObjects[i];
                var curStrainData = new Note(curHitOb.Lane - 1, (int)(curHitOb.StartTime / rate),
                    curHitOb.IsLongNote ? (int)(curHitOb.EndTime / rate) : -1);
                // Add Strain Solver Data to list
                notes.Add(curStrainData);
            }

            MACalculator.Calculate(notes, Map.GetKeyCount(), out var difficulty, out var list);
            OverallDifficulty = (float)difficulty;
            StrainSolverData = list.Select(data =>
                    new StrainSolverData(
                        (float)data.Time,
                        (float)data.Time,
                        (float)MACalculator.StarToLevel(data.D)
                    )
                )
                .ToList();
        }

        /// <summary>
        ///     Checks to see if the map rating is inacurrate due to vibro/rolls
        /// </summary>
        private void ComputeForPatternFlags()
        {
            // If 10% or more of the map has longjack manip, flag it as vibro map
            if (VibroInaccuracyConfidence / StrainSolverData.Count > 0.10)
                QssPatternFlags |= QssPatternFlags.SimpleVibro;

            // If 15% or more of the map has roll manip, flag it as roll map
            if (RollInaccuracyConfidence / StrainSolverData.Count > 0.15)
                QssPatternFlags |= QssPatternFlags.Rolls;
        }
    }
}