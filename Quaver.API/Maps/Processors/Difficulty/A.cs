using System;
using System.Collections.Generic;
using System.Linq;

namespace Quaver.API.Maps.Processors.Difficulty
{
    public struct Note
    {
        public readonly int Column; // key/column index
        public readonly int Head; // head (hit) time
        public readonly int Tail; // tail time (or -1 if not LN)

        public Note(int column, int head, int tail)
        {
            Column = column;
            Head = head;
            Tail = tail;
        }
    }

    /// <summary>
    /// Used to store various computed values at a corner (time point).
    /// </summary>
    public class CornerData
    {
        public double Time;
        public double Jbar;
        public double Xbar;
        public double Pbar;
        public double Abar;
        public double Rbar;
        public double C;
        public double Ks;
        public double D;
        public double Weight;
    }

    /// <summary>
    /// MACalculator computes the difficulty rating from noteSeq (a sequence of notes).
    /// All methods are static.
    /// </summary>
    public static class MACalculator
    {
        /// <param name="noteSeq">List of Note objects.</param>
        /// <param name="keyCount">Number of keys (columns).</param>
        /// <returns>The computed difficulty (Level) as an int.</returns>
        public static void Calculate(List<Note> noteSeq, int keyCount, out double difficulty, out List<CornerData> cornerDataList)
        {
            // Fixed tuning constants.
            const double lambdaN = 5;
            const double lambda1 = 0.11;
            const double lambda3 = 24.0;
            const double lambda2 = 6.8; // For Malody
            // Malody has no lambda4 for releases
            const double w0 = 0.4;
            const double w1 = 2.7;
            const double p1 = 1.5;
            const double w2 = 0.27;
            const double p0 = 1.0;
            const double x = 0.085; // for Malody

            // --- Sort notes by head time, then by column ---
            noteSeq.Sort((a, b) =>
            {
                var cmp = a.Head.CompareTo(b.Head);
                return cmp == 0 ? a.Column.CompareTo(b.Column) : cmp;
            });

            // --- Group notes by column ---
            var noteDict = new Dictionary<int, List<Note>>();
            foreach (var note in noteSeq)
            {
                if (!noteDict.ContainsKey(note.Column))
                    noteDict[note.Column] = new List<Note>();
                noteDict[note.Column].Add(note);
            }

            var noteSeqByColumn = noteDict
                .OrderBy(kvp => kvp.Key)
                .Select(kvp => kvp.Value)
                .ToList();

            // --- Long notes ---
            var lnSeq = noteSeq.Where(n => n.Tail >= 0).ToList();
            var tailSeq = lnSeq.OrderBy(n => n.Tail).ToList();

            var lnDict = new Dictionary<int, List<Note>>();
            foreach (var note in lnSeq)
            {
                if (!lnDict.ContainsKey(note.Column))
                    lnDict[note.Column] = new List<Note>();
                lnDict[note.Column].Add(note);
            }

            var lnSeqByColumn = lnDict
                .OrderBy(kvp => kvp.Key)
                .Select(kvp => kvp.Value)
                .ToList();

            var maxHead = noteSeq.Max(n => n.Head);
            var maxTail = noteSeq.Max(n => n.Tail);
            var T = Math.Max(maxHead, maxTail) + 1;

            // --- Determine Corner Times for base variables and for A ---
            var cornersBase = new HashSet<int>();
            foreach (var note in noteSeq)
            {
                cornersBase.Add(note.Head);
                if (note.Tail >= 0)
                    cornersBase.Add(note.Tail);
            }

            foreach (var s in cornersBase.ToList())
            {
                cornersBase.Add(s + 501);
                cornersBase.Add(s - 499);
                cornersBase.Add(s + 1);
            }

            cornersBase.Add(0);
            cornersBase.Add(T);
            var cornersBaseList = cornersBase.Where(s => s >= 0 && s <= T).ToList();
            cornersBaseList.Sort();

            var cornersA = new HashSet<int>();
            foreach (var note in noteSeq)
            {
                cornersA.Add(note.Head);
                if (note.Tail >= 0)
                    cornersA.Add(note.Tail);
            }

            foreach (var s in cornersA.ToList())
            {
                cornersA.Add(s + 1000);
                cornersA.Add(s - 1000);
            }

            cornersA.Add(0);
            cornersA.Add(T);
            var cornersAList = cornersA.Where(s => s >= 0 && s <= T).ToList();
            cornersAList.Sort();

            var allCornersSet = new HashSet<int>(cornersBaseList);
            allCornersSet.UnionWith(cornersAList);
            var allCornersList = allCornersSet.ToList();
            allCornersList.Sort();

            var allCorners = allCornersList.Select(val => (double)val).ToArray();
            var baseCorners = cornersBaseList.Select(val => (double)val).ToArray();
            var aCorners = cornersAList.Select(val => (double)val).ToArray();

            // --- Section 2.3: Compute Jbar ---
            // Console.WriteLine("2.3");

            // Allocate arrays for each column. For each column k,
            // J_ks[k] will store the unsmoothed J values on baseCorners,
            // and delta_ks[k] will store the corresponding delta values.
            var jKs = new Dictionary<int, double[]>();
            var deltaKs = new Dictionary<int, double[]>();
            for (var k = 0; k < keyCount; k++)
            {
                jKs[k] = new double[baseCorners.Length];
                deltaKs[k] = new double[baseCorners.Length];
                // Initialize delta_ks to a large value.
                for (var j = 0; j < baseCorners.Length; j++)
                {
                    deltaKs[k][j] = 1e9;
                }
            }

            // For each column, compute unsmoothed J using a linear sweep over baseCorners.
            for (var k = 0; k < keyCount; k++)
            {
                var notes = noteSeqByColumn[k];
                var pointer = 0; // pointer over the baseCorners array

                // For each adjacent note pair in the column.
                for (var i = 0; i < notes.Count - 1; i++)
                {
                    var start = notes[i].Head;
                    var end = notes[i + 1].Head;
                    var delta = 0.001 * (end - start);
                    var val = (1.0 / delta) * (1.0 / (delta + lambda1 * Math.Pow(x, 0.25)));
                    var jVal = val * JackNerfer(delta);

                    // Advance pointer until we reach the first base corner >= start.
                    while (pointer < baseCorners.Length && baseCorners[pointer] < start)
                    {
                        pointer++;
                    }

                    // For all base corners in [start, end), assign J_val and delta.
                    while (pointer < baseCorners.Length && baseCorners[pointer] < end)
                    {
                        jKs[k][pointer] = jVal;
                        deltaKs[k][pointer] = delta;
                        pointer++;
                    }
                }
            }

            // Smooth each column’s J using a sliding ±500 window.
            var jbarKs = new Dictionary<int, double[]>();
            for (var k = 0; k < keyCount; k++)
            {
                jbarKs[k] = SmoothOnCorners(baseCorners, jKs[k], 500, 0.001, "sum");
            }

            // Now, for each base corner, aggregate across columns using the lambda_n–power average.
            var jbarBase = new double[baseCorners.Length];
            for (var j = 0; j < baseCorners.Length; j++)
            {
                double num = 0.0, den = 0.0;
                for (var k = 0; k < keyCount; k++)
                {
                    var v = Math.Max(jbarKs[k][j], 0);
                    var weight = 1.0 / deltaKs[k][j];
                    num += Math.Pow(v, lambdaN) * weight;
                    den += weight;
                }

                var avg = num / Math.Max(1e-9, den);
                jbarBase[j] = Math.Pow(avg, 1.0 / lambdaN);
            }

            // Interpolate Jbar from baseCorners to allCorners.
            var jbar = InterpValues(allCorners, baseCorners, jbarBase);


            // --- Section 2.4: Compute Xbar ---
            // Console.WriteLine("2.4");
            var crossMatrix = new List<List<double>>()
            {
                new List<double> { -1 },
                new List<double> { 0.075, 0.075 },
                new List<double> { 0.125, 0.05, 0.125 },
                new List<double> { 0.125, 0.125, 0.125, 0.125 },
                new List<double> { 0.175, 0.25, 0.05, 0.25, 0.175 },
                new List<double> { 0.175, 0.25, 0.175, 0.175, 0.25, 0.175 },
                new List<double> { 0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225 },
                new List<double> { 0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225 },
                new List<double> { 0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275 },
                new List<double> { 0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275 },
                new List<double> { 0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325 }
            };

            // Allocate an array for each key pair (for k=0..keyCount, total keyCount+1 arrays).
            var xKs = new Dictionary<int, double[]>();
            for (var k = 0; k <= keyCount; k++)
            {
                xKs[k] = new double[baseCorners.Length]; // All values default to 0.
            }

            // For each k, compute a step function over the baseCorners that is constant over each interval.
            // Instead of checking every baseCorner for each interval, we “sweep” through baseCorners with a pointer.
            for (var k = 0; k <= keyCount; k++)
            {
                // Determine the merged note sequence for this key–pair:
                List<Note> notesInPair;
                if (k == 0)
                {
                    notesInPair = noteSeqByColumn[0];
                }
                else if (k == keyCount)
                {
                    notesInPair = noteSeqByColumn[keyCount - 1];
                }
                else
                {
                    // Merge the two sorted lists from columns (k–1) and k.
                    notesInPair = MergeSorted(noteSeqByColumn[k - 1], noteSeqByColumn[k]);
                }

                // pointer scans through baseCorners once.
                var pointer = 0;
                for (var i = 1; i < notesInPair.Count; i++)
                {
                    // For this note pair, define the interval [start, end)
                    var start = notesInPair[i - 1].Head;
                    var end = notesInPair[i].Head;
                    var delta = 0.001 * (end - start);
                    var val = 0.16 * Math.Pow(Math.Max(x, delta), -2);

                    // Advance the pointer until the current base corner is within [start, end)
                    while (pointer < baseCorners.Length && baseCorners[pointer] < start)
                    {
                        pointer++;
                    }

                    // Now assign the value for all baseCorners in [start, end)
                    while (pointer < baseCorners.Length && baseCorners[pointer] < end)
                    {
                        xKs[k][pointer] = val;
                        pointer++;
                    }
                    // Note: Because note intervals are in increasing order, pointer only moves forward.
                }
            }

            // Combine the X_ks values across k using the cross–matrix coefficients.
            // (The cross–matrix for keyCount returns a list of keyCount+1 coefficients.)
            var crossCoeff = crossMatrix[keyCount];
            var xBase = new double[baseCorners.Length];
            for (var i = 0; i < baseCorners.Length; i++)
            {
                var sum = 0.0;
                for (var k = 0; k <= keyCount; k++)
                {
                    sum += xKs[k][i] * crossCoeff[k];
                }

                xBase[i] = sum;
            }

            // Smooth and interpolate as in the rest of the algorithm.
            var xbarBase = SmoothOnCorners(baseCorners, xBase, 500, 0.001, "sum");
            var xbar = InterpValues(allCorners, baseCorners, xbarBase);

            // --- Section 2.5: Compute Pbar ---
            // Console.WriteLine("2.5");

            // Build LN_bodies array over time [0, T)
            var lnBodies = new double[T];
            for (var i = 0; i < T; i++)
                lnBodies[i] = 0.0;

            // For each long note, add contributions in three segments:
            //   from h to t0, add nothing;
            //   from t0 to t1, add 1.3;
            //   from t1 to t, add 1.0.
            foreach (var note in lnSeq)
            {
                var h = note.Head;
                var t = note.Tail;
                var t0 = Math.Min(h + 60, t);
                var t1 = Math.Min(h + 120, t);
                for (var i = t0; i < t1; i++)
                    lnBodies[i] += 1.3;
                for (var i = t1; i < t; i++)
                    lnBodies[i] += 1.0;
            }

            // Compute cumulative sum over LN_bodies
            var cumsumLn = new double[T + 1];
            cumsumLn[0] = 0.0;
            for (var i = 1; i <= T; i++)
            {
                cumsumLn[i] = cumsumLn[i - 1] + lnBodies[i - 1];
            }

            // LN_sum returns the exact sum over LN_bodies in the interval [a, b)

            // Stream Booster

            // Allocate P_step on the base grid.
            var pStep = new double[baseCorners.Length];
            for (var i = 0; i < baseCorners.Length; i++)
                pStep[i] = 0.0;

            // Process each adjacent pair of notes in noteSeq. Since noteSeq is sorted by head time,
            // the interval [h_l, h_r) for each pair will also be in increasing order.
            // We maintain a pointer into baseCorners that advances monotonically.
            var pointerP = 0;
            for (var i = 0; i < noteSeq.Count - 1; i++)
            {
                var hL = noteSeq[i].Head;
                var hR = noteSeq[i + 1].Head;
                double deltaTime = hR - hL;

                if (deltaTime < 1e-9)
                {
                    // Handle Dirac delta spikes when consecutive notes have identical times.
                    // Find the base corner exactly equal to h_l (using binary search is acceptable here)
                    var idx = Array.BinarySearch(baseCorners, hL);
                    if (idx < 0)
                        idx = ~idx;
                    if (idx < baseCorners.Length && Math.Abs(baseCorners[idx] - hL) < 1e-9)
                    {
                        var spike = 1000 * Math.Pow(0.02 * (4 / x - lambda3), 0.25);
                        pStep[idx] += spike;
                    }

                    continue;
                }

                // Compute constant values for this interval.
                var delta = 0.001 * deltaTime;
                var v = 1 + lambda2 * 0.001 * LnSum(hL, hR, cumsumLn);
                var bVal = StreamBooster(delta);
                double inc;
                if (delta < 2 * x / 3)
                {
                    inc = (1.0 / delta) * Math.Pow(0.08 * (1.0 / x) *
                                                   (1 - lambda3 * (1.0 / x) * Math.Pow(delta - x / 2, 2)), 0.25) *
                          bVal * v;
                }
                else
                {
                    inc = (1.0 / delta) * Math.Pow(0.08 * (1.0 / x) *
                                                   (1 - lambda3 * (1.0 / x) * Math.Pow(x / 6, 2)), 0.25) * bVal * v;
                }

                // Advance pointerP until the current base corner is at least h_l.
                while (pointerP < baseCorners.Length && baseCorners[pointerP] < hL)
                    pointerP++;

                // For every base corner in the interval [h_l, h_r), add the increment.
                while (pointerP < baseCorners.Length && baseCorners[pointerP] < hR)
                {
                    pStep[pointerP] += inc;
                    pointerP++;
                }
                // Since noteSeq is sorted, pointerP never resets backwards.
            }

            // Smooth and interpolate P_step as before.
            var pbarBase = SmoothOnCorners(baseCorners, pStep, 500, 0.001, "sum");
            var pbar = InterpValues(allCorners, baseCorners, pbarBase);


            // --- Section 2.6: Compute Abar ---
            // Console.WriteLine("2.6");

            // Allocate a boolean active–flag array per key over baseCorners.
            // (New bool arrays are false by default; no need to loop and set to false.)
            var kuKs = new Dictionary<int, bool[]>();
            for (var k = 0; k < keyCount; k++)
            {
                kuKs[k] = new bool[baseCorners.Length];
            }

            // For each key, mark baseCorners that lie within the “active” interval for each note.
            // The active interval for a note is [max(note.Head - 150, 0), (note.Tail < 0 ? note.Head + 150 : min(note.Tail + 150, T - 1))).
            for (var k = 0; k < keyCount; k++)
            {
                // Get the note sequence for key k.
                var notes = noteSeqByColumn[k];
                foreach (var note in notes)
                {
                    var activeStart = Math.Max(note.Head - 150, 0);
                    var activeEnd = (note.Tail < 0) ? (note.Head + 150) : Math.Min(note.Tail + 150, T - 1);
                    // Use binary search to find the first baseCorner >= activeStart.
                    var startIdx = Array.BinarySearch(baseCorners, activeStart);
                    if (startIdx < 0)
                        startIdx = ~startIdx;
                    // Advance pointer until the base corner is no longer less than activeEnd.
                    var idx = startIdx;
                    while (idx < baseCorners.Length && baseCorners[idx] < activeEnd)
                    {
                        kuKs[k][idx] = true;
                        idx++;
                    }
                }
            }

            // For each baseCorner, build a list of active keys.
            var kuSCols = new List<List<int>>(baseCorners.Length);
            for (var i = 0; i < baseCorners.Length; i++)
            {
                var activeCols = new List<int>();
                for (var k = 0; k < keyCount; k++)
                {
                    if (kuKs[k][i])
                        activeCols.Add(k);
                }

                kuSCols.Add(activeCols);
            }

            // Compute dks: For each baseCorner, for each adjacent pair of active keys,
            // compute the difference measure: |delta_ks[k0] - delta_ks[k1]| + max(0, max(delta_ks[k0], delta_ks[k1]) - 0.3).
            var dks = new Dictionary<int, double[]>();
            for (var k = 0; k < keyCount - 1; k++)
            {
                dks[k] = new double[baseCorners.Length];
            }

            for (var i = 0; i < baseCorners.Length; i++)
            {
                var cols = kuSCols[i];
                // Only if there are at least two active keys.
                for (var j = 0; j < cols.Count - 1; j++)
                {
                    var k0 = cols[j];
                    var k1 = cols[j + 1];
                    dks[k0][i] = Math.Abs(deltaKs[k0][i] - deltaKs[k1][i]) +
                                 0.4 * Math.Max(0, Math.Max(deltaKs[k0][i], deltaKs[k1][i]) - 0.11);
                }
            }

            // Compute A_step on the A–grid (ACorners). Start with a default value of 1.0.
            var aStep = new double[aCorners.Length];
            for (var i = 0; i < aCorners.Length; i++)
                aStep[i] = 1.0;

            // For each A–corner, determine the corresponding value from the base grid.
            // We do this by finding the nearest baseCorner using binary search, then using the active key list there.
            for (var i = 0; i < aCorners.Length; i++)
            {
                var s = aCorners[i];
                var idx = Array.BinarySearch(baseCorners, s);
                if (idx < 0)
                    idx = ~idx;
                if (idx >= baseCorners.Length)
                    idx = baseCorners.Length - 1;
                // Get the list of active keys at this base corner.
                var cols = kuSCols[idx];
                // For each adjacent pair of active keys in that list, adjust A_step.
                for (var j = 0; j < cols.Count - 1; j++)
                {
                    var k0 = cols[j];
                    var k1 = cols[j + 1];
                    var dVal = dks[k0][idx];
                    if (dVal < 0.02)
                        aStep[i] *= Math.Min(0.75 + 0.5 * Math.Max(deltaKs[k0][idx], deltaKs[k1][idx]), 1);
                    else if (dVal < 0.07)
                        aStep[i] *= Math.Min(0.65 + 5 * dVal + 0.5 * Math.Max(deltaKs[k0][idx], deltaKs[k1][idx]),
                            1);
                }
            }

            // Finally, smooth A_step on the ACorners with a ±500 window (using average smoothing),
            // then interpolate Abar from the ACorners to the overall grid.
            var abarA = SmoothOnCorners(aCorners, aStep, 500, 1.0, "avg");
            var abar = InterpValues(allCorners, aCorners, abarA);


            // --- Section 2.7: Compute Rbar ---
            // Console.WriteLine("2.7");

            var rbar = new double[allCorners.Length];
            for (var i = 0; i < rbar.Length; i++)
            {
                rbar[i] = 0;
            }


            // --- Section 3: Compute C and Ks ---
            // Console.WriteLine("3");
            var noteHitTimes = noteSeq.Select(n => n.Head).ToList();
            noteHitTimes.Sort();
            var cStep = new double[baseCorners.Length];
            for (var i = 0; i < baseCorners.Length; i++)
            {
                var s = baseCorners[i];
                var low = s - 500;
                var high = s + 500;
                var cntLow = LowerBound(noteHitTimes, (int)low);
                var cntHigh = LowerBound(noteHitTimes, (int)high);
                var cnt = cntHigh - cntLow;
                cStep[i] = cnt;
            }

            var cBase = cStep;
            var cArr = StepInterp(allCorners, baseCorners, cBase);

            var ksStep = new double[baseCorners.Length];
            for (var i = 0; i < baseCorners.Length; i++)
            {
                var cntActive = 0;
                for (var k = 0; k < keyCount; k++)
                {
                    if (kuKs[k][i])
                        cntActive++;
                }

                ksStep[i] = Math.Max(cntActive, 1);
            }

            var ksBase = ksStep;
            var ksArr = StepInterp(allCorners, baseCorners, ksBase);

            // --- Final Computations: Compute S, T, D ---
            var N = allCorners.Length;
            var sAll = new double[N];
            var all = new double[N];
            var dAll = new double[N];
            for (var i = 0; i < N; i++)
            {
                var aVal = abar[i];
                var jVal = jbar[i];
                var xVal = xbar[i];
                var pVal = pbar[i];
                var rVal = rbar[i];
                var ksVal = ksArr[i];

                var term1 = Math.Pow(Math.Pow(aVal, 3.0 / ksVal) * jVal, 1.5);
                var term2 = Math.Pow(Math.Pow(aVal, 2.0 / 3.0) * (0.8 * pVal + rVal), 1.5);
                var sVal = Math.Pow(w0 * term1 + (1 - w0) * term2, 2.0 / 3.0);
                sAll[i] = sVal;
                var val = (Math.Pow(aVal, 3.0 / ksVal) * xVal) / (xVal + sVal + 1);
                all[i] = val;
                dAll[i] = w1 * Math.Pow(sVal, 0.5) * Math.Pow(val, p1) + sVal * w2;
            }

            // --- Weighted–Percentile Calculation ---
            var gaps = new double[N];
            if (N == 1)
                gaps[0] = 0;
            else
            {
                gaps[0] = (allCorners[1] - allCorners[0]) / 2.0;
                gaps[N - 1] = (allCorners[N - 1] - allCorners[N - 2]) / 2.0;
                for (var i = 1; i < N - 1; i++)
                    gaps[i] = (allCorners[i + 1] - allCorners[i - 1]) / 2.0;
            }

            var effectiveWeights = new double[N];
            for (var i = 0; i < N; i++)
                effectiveWeights[i] = cArr[i] * gaps[i];

            cornerDataList = new List<CornerData>();
            for (var i = 0; i < N; i++)
            {
                cornerDataList.Add(new CornerData
                {
                    Time = allCorners[i],
                    Jbar = jbar[i],
                    Xbar = xbar[i],
                    Pbar = pbar[i],
                    Abar = abar[i],
                    Rbar = rbar[i],
                    C = cArr[i],
                    Ks = ksArr[i],
                    D = dAll[i],
                    Weight = effectiveWeights[i]
                });
            }

            var sortedList = cornerDataList.OrderBy(cd => cd.D).ToList();
            var dSorted = sortedList.Select(cd => cd.D).ToArray();
            var cumWeights = new double[sortedList.Count];
            var sumW = 0.0;
            for (var i = 0; i < sortedList.Count; i++)
            {
                sumW += sortedList[i].Weight;
                cumWeights[i] = sumW;
            }

            var totalWeight = sumW;
            var normCumWeights = cumWeights.Select(cw => cw / totalWeight).ToArray();

            var targetPercentiles = new[] { 0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815 };
            var indices = new List<int>();
            foreach (var tp in targetPercentiles)
            {
                var idx = Array.FindIndex(normCumWeights, cw => cw >= tp);
                if (idx < 0)
                    idx = sortedList.Count - 1;
                indices.Add(idx);
            }

            double percentile93, percentile83;
            if (indices.Count >= 8)
            {
                var sum93 = 0.0;
                for (var i = 0; i < 4; i++)
                    sum93 += sortedList[indices[i]].D;
                percentile93 = sum93 / 4.0;
                var sum83 = 0.0;
                for (var i = 4; i < 8; i++)
                    sum83 += sortedList[indices[i]].D;
                percentile83 = sum83 / 4.0;
            }
            else
            {
                percentile93 = sortedList.Average(cd => cd.D);
                percentile83 = percentile93;
            }

            var numWeighted = 0.0;
            var denWeighted = 0.0;
            for (var i = 0; i < sortedList.Count; i++)
            {
                numWeighted += Math.Pow(sortedList[i].D, lambdaN) * sortedList[i].Weight;
                denWeighted += sortedList[i].Weight;
            }

            var weightedMean = Math.Pow(numWeighted / denWeighted, 1.0 / lambdaN);
            var sr = (0.88 * percentile93) * 0.25 + (0.94 * percentile83) * 0.2 + weightedMean * 0.55;
            sr = Math.Pow(sr, p0) / Math.Pow(8, p0) * 8;
            var totalNotes = noteSeq.Count + 0.5 * lnSeq.Count;
            sr *= totalNotes / (totalNotes + 60);
            if (sr <= 2)
                sr = Math.Sqrt(sr * 2);

            sr *= 1 - 0.0075 * keyCount;
            difficulty = StarToLevel(sr);
        }

        private static double LnSum(int a, int b, double[] cumsumLn) => cumsumLn[b] - cumsumLn[a];

        private static double StreamBooster(double delta)
        {
            var val = 7.5 / delta;
            if (val > 160 && val < 360) return 1 + 1.7e-7 * (val - 160) * Math.Pow(val - 360, 2);
            return 1.0;
        }

        private static double JackNerfer(double delta)
        {
            return 1 - 7e-5 * Math.Pow(0.15 + Math.Abs(delta - 0.08), -4);
        }

        public static double StarToLevel(double x)
        {
            return x * 5;
        }

        #region Helper Methods

        // Returns the cumulative sum array for f evaluated on x.
        private static double[] CumulativeSum(double[] x, double[] f)
        {
            var n = x.Length;
            var F = new double[n];
            F[0] = 0.0;
            for (var i = 1; i < n; i++)
            {
                F[i] = F[i - 1] + f[i - 1] * (x[i] - x[i - 1]);
            }

            return F;
        }

        // Query cumulative sum at q.
        private static double QueryCumsum(double q, double[] x, double[] F, double[] f)
        {
            if (q <= x[0])
                return 0.0;
            if (q >= x[^1])
                return F[x.Length - 1];
            var idx = Array.BinarySearch(x, q);
            if (idx < 0)
                idx = ~idx;
            var i = idx - 1;
            return F[i] + f[i] * (q - x[i]);
        }

        // Smooth values f (defined on x) over a symmetric window.
        // If mode is "avg", returns the average; otherwise multiplies the integral by scale.
        private static double[] SmoothOnCorners(double[] x, double[] f, double window, double scale, string mode)
        {
            var n = f.Length;
            var F = CumulativeSum(x, f);
            var g = new double[n];
            for (var i = 0; i < n; i++)
            {
                var s = x[i];
                var a = Math.Max(s - window, x[0]);
                var b = Math.Min(s + window, x[^1]);
                var val = QueryCumsum(b, x, F, f) - QueryCumsum(a, x, F, f);
                if (mode == "avg")
                    g[i] = (b - a) > 0 ? val / (b - a) : 0.0;
                else
                    g[i] = scale * val;
            }

            return g;
        }

        // Linear interpolation from old_x, old_vals to new_x.
        private static double[] InterpValues(double[] newX, double[] oldX, double[] oldVals)
        {
            var n = newX.Length;
            var newVals = new double[n];
            for (var i = 0; i < n; i++)
            {
                var xVal = newX[i];
                if (xVal <= oldX[0])
                    newVals[i] = oldVals[0];
                else if (xVal >= oldX[^1])
                    newVals[i] = oldVals[oldX.Length - 1];
                else
                {
                    var idx = Array.BinarySearch(oldX, xVal);
                    if (idx < 0)
                        idx = ~idx;
                    var j = idx - 1;
                    var t = (xVal - oldX[j]) / (oldX[j + 1] - oldX[j]);
                    newVals[i] = oldVals[j] + t * (oldVals[j + 1] - oldVals[j]);
                }
            }

            return newVals;
        }

        // Step–function interpolation (zero–order hold).
        private static double[] StepInterp(double[] newX, double[] oldX, double[] oldVals)
        {
            var n = newX.Length;
            var newVals = new double[n];
            for (var i = 0; i < n; i++)
            {
                var xVal = newX[i];
                var idx = Array.BinarySearch(oldX, xVal);
                if (idx < 0)
                    idx = ~idx;
                idx = idx - 1;
                if (idx < 0)
                    idx = 0;
                if (idx >= oldVals.Length)
                    idx = oldVals.Length - 1;
                newVals[i] = oldVals[idx];
            }

            return newVals;
        }

        // Merges two sorted lists of Note (sorted by Head) into one sorted list.
        private static List<Note> MergeSorted(List<Note> list1, List<Note> list2)
        {
            var merged = new List<Note>();
            int i = 0, j = 0;
            while (i < list1.Count && j < list2.Count)
            {
                if (list1[i].Head <= list2[j].Head)
                {
                    merged.Add(list1[i]);
                    i++;
                }
                else
                {
                    merged.Add(list2[j]);
                    j++;
                }
            }

            while (i < list1.Count)
            {
                merged.Add(list1[i]);
                i++;
            }

            while (j < list2.Count)
            {
                merged.Add(list2[j]);
                j++;
            }

            return merged;
        }

        // Implements lower_bound: first index at which list[index] >= value.
        private static int LowerBound(List<int> list, int value)
        {
            var low = 0;
            var high = list.Count;
            while (low < high)
            {
                var mid = (low + high) / 2;
                if (list[mid] < value)
                    low = mid + 1;
                else
                    high = mid;
            }

            return low;
        }

        #endregion
    }
}